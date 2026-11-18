import os
import glob
import cv2
import numpy as np


class PlateBinarizer:
    """
    Modul Segmentasi Biner, Perbaikan Morfologi, dan Pemisahan Dual-ROI (Tahap 7 - 9).
    Input: Citra grayscale tajam (`sharpened_gray`) yang SUDAH berstandar Background Hitam (Gelap) & Tulisan Putih (Terang).
    Urutan proses sesuai diagram sistem baru:
    - Tahap A (P12 Segmentasi Biner): Thresholding Otsu & Adaptive -> Background Hitam (0) & Karakter Putih (255).
    - Tahap B (P13-P14 Perbaikan Morfologi): Morphological Closing langsung pada piksel putih (255).
    - Tahap C (Segmentasi Dual-ROI): Memisahkan ROI Nomor Utama (`roi_cleaned`) dengan ROI Angka Pajak (`tax_roi_binary` & `tax_roi_gray`).
    """

    def __init__(self, method="otsu", block_size=31, c_val=-5):
        self.method = method
        self.block_size = block_size
        self.c_val = c_val

    def upscale_if_small(self, gray_img, target_height=160):
        h, w = gray_img.shape[:2]
        if 0 < h < target_height:
            scale = float(target_height) / float(h)
            return cv2.resize(gray_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        return gray_img.copy()

    def sharpen(self, gray_img):
        """Materi P8: Unsharp Masking (cadangan jika dipanggil mandiri)."""
        blur = cv2.GaussianBlur(gray_img, (0, 0), 2.0)
        return cv2.addWeighted(gray_img, 1.6, blur, -0.6, 0)

    # ------------------------------------------------------------------
    # TAHAP A: Segmentasi Biner (Latar Hitam = 0, Karakter Putih = 255)
    # ------------------------------------------------------------------
    def _ensure_dark_bg_white_text(self, binary_img):
        """
        Pastikan standar polaritas biner: BACKGROUND HITAM (0) dan KARAKTER PUTIH (255).
        Karena luas latar pelat selalu lebih dominan dibanding luas guratan huruf,
        rasio piksel putih (255) pada pelat normal berada di bawah 50%.
        Jika piksel putih > 52%, berarti latar masih putih -> lakukan inversi.
        """
        white_fraction = float(np.sum(binary_img == 255)) / float(binary_img.size)
        if white_fraction > 0.52:
            return cv2.bitwise_not(binary_img)
        return binary_img.copy()

    def _threshold(self, gray_img, mode):
        if mode == "adaptive":
            b = self.block_size if self.block_size % 2 == 1 else self.block_size + 1
            binary = cv2.adaptiveThreshold(
                gray_img, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                b, self.c_val
            )
        else:
            _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return self._ensure_dark_bg_white_text(binary)

    # ------------------------------------------------------------------
    # TAHAP B: Perbaikan Morfologi (Morphological Closing pada Piksel Putih 255)
    # ------------------------------------------------------------------
    def _close_gaps(self, binary_img):
        """
        Materi P13-P14: Morphological Closing langsung pada piksel karakter putih (255)
        untuk menyambung celah huruf/angka yang terputus akibat goresan, baut, atau tali tipis.
        """
        h = binary_img.shape[0]
        k = max(3, (h // 32) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

    # ------------------------------------------------------------------
    # TAHAP C: Segmentasi Dual-ROI (Memisahkan Nomor Utama & Angka Pajak MM.YY)
    # ------------------------------------------------------------------
    def _clean_main_components(self, morphed_bin):
        """
        Mengekstrak hanya karakter baris utama pada citra biner (latar 0, teks 255):
        - Memotong sedikit tepi bingkai luar pelat,
        - Menggunakan connectedComponentsWithStats pada piksel putih (255),
        - Membuang bingkai panjang, bintik baut kecil, serta membuang angka pajak di bawah
          (komponen dengan tinggi < 48% tinggi karakter utama atau titik tengah y > 70% tinggi pelat).
        """
        cleaned = morphed_bin.copy()
        h, w = cleaned.shape[:2]

        # Bersihkan tepi bingkai luar (2% kiri-kanan, 3% atas, 6% bawah)
        by1, by2 = int(h * 0.03), int(h * 0.94)
        bx1, bx2 = int(w * 0.02), int(w * 0.98)
        border_mask = np.zeros_like(cleaned)
        border_mask[by1:by2, bx1:bx2] = 255
        cleaned = cv2.bitwise_and(cleaned, border_mask)

        n, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        if n <= 1:
            return cleaned, 0

        min_area = max(20, int(h * w * 0.0015))
        # Cari tinggi karakter utama dari kandidat di paruh atas-tengah pelat
        main_candidates = [
            stats[i][3] for i in range(1, n)
            if stats[i][4] >= min_area
            and stats[i][2] <= 0.45 * w
            and centroids[i][1] <= 0.72 * h
        ]
        max_ch = max(main_candidates) if main_candidates else h

        removed_blobs = 0
        for i in range(1, n):
            x, y, cw, ch, area = stats[i]
            yc = centroids[i][1]
            # Kriteria komponen yang dihapus (diubah menjadi 0 / hitam):
            # 1. Terlalu lebar (> 48% lebar pelat -> garis bingkai)
            # 2. Terlalu kecil (< min_area -> noise/baut)
            # 3. Terlalu pendek (< 48% tinggi huruf utama -> angka pajak)
            # 4. Berada di area bawah (yc > 70% tinggi pelat dan ch < 65% max_ch -> baris pajak)
            if cw > 0.48 * w or area < min_area or ch < 0.48 * max_ch or (yc > 0.70 * h and ch < 0.65 * max_ch):
                cleaned[labels == i] = 0
                removed_blobs += 1

        return cleaned, removed_blobs

    def _pad(self, img, value=0):
        return cv2.copyMakeBorder(img, 15, 15, 20, 20, cv2.BORDER_CONSTANT, value=value)

    def separate_main_and_tax_roi(self, morphed_bin, sharpened_gray):
        """
        [KEBARUAN 2] Memisahkan citra pelat menjadi 2 ROI:
        1. `roi_cleaned`: Citra biner Nomor Utama (tanpa angka pajak & tanpa bingkai), diberi padding hitam (0).
        2. `tax_roi_binary` & `tax_roi_gray`: Sub-area tengah-bawah pelat (y: 62%-93%, x: 22%-78%)
           tempat terletaknya angka bulan & tahun pajak (MM.YY, contoh '01.28'), di-upscale 2x
           dan dibersihkan dari noise kecil tanpa membawa teks iklan dudukan plat di pinggir bawah.
        """
        h, w = morphed_bin.shape[:2]

        # 1. Ekstrak ROI Nomor Utama
        main_clean, removed_blobs = self._clean_main_components(morphed_bin)
        roi_cleaned = self._pad(main_clean, value=0)

        # 2. Ekstrak ROI Angka Pajak (tengah-bawah pelat: y 62%-93%, x 22%-78%)
        y1, y2 = int(h * 0.62), int(h * 0.93)
        x1, x2 = int(w * 0.22), int(w * 0.78)

        tax_gray_crop = sharpened_gray[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else sharpened_gray.copy()
        if tax_gray_crop.size == 0:
            tax_gray_crop = sharpened_gray.copy()

        # Upscale 2x pada ROI pajak agar digit kecil lebih jelas dibaca EasyOCR
        tax_roi_gray = cv2.resize(tax_gray_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        _, tax_bin_raw = cv2.threshold(tax_roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        tax_bin = self._ensure_dark_bg_white_text(tax_bin_raw)

        # Bersihkan bintik noise sangat kecil pada ROI pajak
        th_h, th_w = tax_bin.shape[:2]
        n_t, labels_t, stats_t, _ = cv2.connectedComponentsWithStats(tax_bin, connectivity=8)
        min_tax_area = max(10, int(th_h * th_w * 0.002))
        for idx in range(1, n_t):
            cw_t, ch_t, area_t = stats_t[idx][2], stats_t[idx][3], stats_t[idx][4]
            if area_t < min_tax_area or cw_t > 0.65 * th_w or ch_t < 0.22 * th_h:
                tax_bin[labels_t == idx] = 0

        tax_roi_binary = cv2.copyMakeBorder(tax_bin, 8, 8, 12, 12, cv2.BORDER_CONSTANT, value=0)

        return {
            "roi_cleaned": roi_cleaned,
            "removed_blobs": removed_blobs,
            "tax_roi_binary": tax_roi_binary,
            "tax_roi_gray": tax_roi_gray,
        }

    # ------------------------------------------------------------------
    # Pipeline Utama PlateBinarizer
    # ------------------------------------------------------------------
    def process(self, sharpened_gray):
        """
        Menjalankan Tahap A (Thresholding) -> Tahap B (Morfologi) -> Tahap C (Pemisahan Dual-ROI).
        """
        if sharpened_gray is None or sharpened_gray.size == 0:
            raise ValueError("Input sharpened_gray kosong!")

        if len(sharpened_gray.shape) == 3:
            sharpened_gray = cv2.cvtColor(sharpened_gray, cv2.COLOR_BGR2GRAY)

        upscaled = self.upscale_if_small(sharpened_gray)

        primary_mode = "adaptive" if self.method == "adaptive" else "otsu"
        secondary_mode = "otsu" if primary_mode == "adaptive" else "adaptive"

        # Tahap A: Segmentasi Biner (Latar Hitam = 0, Karakter Putih = 255)
        binary_raw = self._threshold(upscaled, primary_mode)
        white_ratio = round(float(np.mean(binary_raw == 255) * 100.0), 1)

        # Tahap B: Perbaikan Morfologi
        morphed = self._close_gaps(binary_raw)

        # Tahap C: Segmentasi Dual-ROI (Nomor Utama & Pajak)
        dual_roi = self.separate_main_and_tax_roi(morphed, upscaled)
        roi_cleaned = dual_roi["roi_cleaned"]

        # Bangun varian sekunder (Adaptive / Otsu alternatif) untuk voting OCR
        bin_sec_raw = self._threshold(upscaled, secondary_mode)
        morphed_sec = self._close_gaps(bin_sec_raw)
        roi_sec, _ = self._clean_main_components(morphed_sec)
        roi_sec_padded = self._pad(roi_sec, value=0)

        # Grayscale baris utama untuk kandidat ke-3 OCR
        h, w = upscaled.shape[:2]
        main_gray_crop = upscaled[int(h * 0.04):int(h * 0.68), int(w * 0.02):int(w * 0.98)]
        if main_gray_crop.size == 0:
            main_gray_crop = upscaled
        upscaled_gray_padded = cv2.copyMakeBorder(main_gray_crop, 15, 15, 20, 20, cv2.BORDER_REPLICATE)

        return {
            "upscaled_gray": upscaled_gray_padded,
            "binary_raw": binary_raw,
            "morphed": morphed,
            "binary_normalized": roi_cleaned,
            "roi_cleaned": roi_cleaned,
            "removed_blobs": dual_roi["removed_blobs"],
            "white_ratio": white_ratio,
            "tax_roi_binary": dual_roi["tax_roi_binary"],
            "tax_roi_gray": dual_roi["tax_roi_gray"],
            "variants": [roi_sec_padded, upscaled_gray_padded]
        }


def resolve_test_vehicle_image(base_dir):
    """
    Prioritas 1: Akses dataset via Kaggle API (`kagglehub.dataset_download`).
    Prioritas 2 (Fallback): Gunakan direktori dataset lokal jika API gagal / offline.
    """
    try:
        import kagglehub
        print("[Dataset] Mencoba mengakses dataset via Kaggle API (kagglehub)...")
        api_path = kagglehub.dataset_download("juanthomaswijaya/indonesian-license-plate-dataset")
        found = sorted(glob.glob(os.path.join(api_path, "**", "images", "test", "*.jpg"), recursive=True))
        found = [f for f in found if "Recognition" not in f]
        if found:
            print(f"[Dataset] Berhasil mengakses via Kaggle API: {found[0]}")
            return found[0]
    except Exception as e:
        print(f"[Dataset] Akses Kaggle API dilewati ({e}). Beralih ke dataset lokal...")

    local_candidates = [
        os.path.join(base_dir, "Dataset", "images", "Indonesian License Plate Dataset", "images", "test"),
        os.path.join(base_dir, "Dataset", "Raw", "Indonesian License Plate Dataset", "images", "test"),
        os.path.join(base_dir, "Output", "01_cropped_plates"),
    ]
    for folder in local_candidates:
        if os.path.exists(folder):
            imgs = sorted(glob.glob(os.path.join(folder, "*.jpg")))
            if imgs:
                return imgs[0]
    return None


if __name__ == "__main__":
    import sys
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    from Script.preprocessor import PlatePreprocessor

    out_dir = os.path.join(base_dir, "Output", "03_binarized")
    os.makedirs(out_dir, exist_ok=True)

    sample_img_path = sys.argv[1] if len(sys.argv) > 1 else resolve_test_vehicle_image(base_dir)
    if not sample_img_path or not os.path.exists(sample_img_path):
        print("Gambar uji tidak ditemukan di Kaggle API maupun lokal.")
        sys.exit(1)

    print(f"Menguji integrasi Hulu -> Tengah pada: {sample_img_path}")
    raw_bgr = cv2.imread(sample_img_path)

    # Coba panggil PlateDetector milik Zulfan terlebih dahulu
    plates_to_process = []
    try:
        from Script.detector import PlateDetector
        det_results = PlateDetector().detect(raw_bgr)
        if det_results:
            for d in det_results:
                plates_to_process.append(d.get("polarity_std", d["crop"]))
    except Exception as e:
        print(f"[Info] PlateDetector dilewati ({e}), menggunakan gambar langsung.")

    if not plates_to_process:
        # Jika gambar sudah berupa close-up pelat (atau model best.pt belum ada), cek polaritas sederhana
        gray_check = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
        polarity_input = cv2.bitwise_not(raw_bgr) if np.median(gray_check) > 127 else raw_bgr
        plates_to_process.append(polarity_input)

    preprocessor = PlatePreprocessor()
    binarizer = PlateBinarizer()

    for idx, pol_img in enumerate(plates_to_process, start=1):
        prep_out = preprocessor.process(pol_img)
        bin_out = binarizer.process(prep_out["sharpened"])

        main_save = os.path.join(out_dir, f"roi_cleaned_{idx}.jpg")
        tax_save = os.path.join(out_dir, f"tax_roi_binary_{idx}.jpg")
        cv2.imwrite(main_save, bin_out["roi_cleaned"])
        cv2.imwrite(tax_save, bin_out["tax_roi_binary"])

        print(f"--- Hasil Pelat #{idx} ---")
        print(f"Rasio Piksel Putih (Biner): {bin_out['white_ratio']}%")
        print(f"Komponen Noise/Pajak Dibuang: {bin_out['removed_blobs']} blob")
        print(f"ROI Nomor Utama disimpan ke : {main_save}")
        print(f"ROI Nomor Pajak disimpan ke : {tax_save}")
