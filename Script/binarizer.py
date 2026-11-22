import sys
import cv2
import numpy as np


class PlateBinarizer:
    """
    Modul Segmentasi Biner, Morfologi & Dual-ROI (Tahap 7-9).
    Input: `sharpened` (Grayscale dengan latar gelap & tulisan terang).
    Standar polaritas biner: Latar Hitam (0) & Karakter Putih (255).
    """

    def __init__(self, method="otsu", block_size=31, c_val=-10):
        self.method = method
        self.block_size = block_size
        self.c_val = c_val

    def _tophat_assist(self, gray_img):
        """
        Bantu pisahkan karakter terang dari latar dengan pencahayaan tidak merata
        menggunakan White Top-Hat morfologi sebelum thresholding.
        """
        h = gray_img.shape[0]
        k = max(15, int(h * 0.42) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        tophat = cv2.morphologyEx(gray_img, cv2.MORPH_TOPHAT, kernel)
        # Gabungkan top-hat dengan citra asli agar goresan tebal tetap kuat
        blended = cv2.addWeighted(gray_img, 0.45, tophat, 0.55, 0)
        return cv2.normalize(blended, None, 0, 255, cv2.NORM_MINMAX)

    def _threshold(self, gray_img, mode):
        """Tahap 7: Segmentasi Biner (Otsu / Adaptive) berstandar latar 0 & teks 255."""
        h = gray_img.shape[0]
        if mode == "adaptive":
            b = max(self.block_size, int(h * 0.42) | 1)
            if b % 2 == 0:
                b += 1
            binary = cv2.adaptiveThreshold(
                gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, b, self.c_val,
            )
        else:
            assisted = self._tophat_assist(gray_img)
            _, binary = cv2.threshold(assisted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Pastikan piksel latar (mayoritas) bernilai 0 dan karakter bernilai 255
        if np.count_nonzero(binary) > binary.size * 0.55:
            binary = cv2.bitwise_not(binary)
        return binary

    def _close_gaps(self, binary_raw):
        """
        Tahap 8: Morphological Closing ringan (3x3) pada karakter putih (255)
        untuk menyambung celah mikro tanpa menutup lubang huruf (8, 0, B, A, 4)
        atau menempelkan pantulan cahaya ke huruf.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.morphologyEx(binary_raw, cv2.MORPH_CLOSE, kernel)

    def _clean_main_components(self, morphed_bin):
        """
        Ekstrak ROI Nomor Utama (Tahap 9a): hapus bingkai luar, bintik noise,
        sisa garis tepi vertikal, dan angka pajak kecil di baris bawah
        menggunakan Connected Components berbasis median tinggi karakter.
        """
        cleaned = morphed_bin.copy()
        h, w = cleaned.shape[:2]
        n, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        if n <= 1:
            return self._pad(cleaned, value=0), 0

        min_area = max(16, int(h * w * 0.0010))
        candidates = [
            stats[i][3] for i in range(1, n)
            if stats[i][4] >= min_area
            and 0.24 * h <= stats[i][3] <= 0.90 * h
            and stats[i][2] <= 0.35 * w
            and stats[i][1] < 0.72 * h
        ]
        if candidates:
            ref_ch = float(np.median(candidates))
        else:
            ref_ch = float(h * 0.52)

        removed = 0
        edge_margin = max(2, int(w * 0.015))
        for i in range(1, n):
            x, y, cw, ch, area = stats[i]
            touches_lr = (x <= edge_margin) or (x + cw >= w - edge_margin)
            touches_top = y <= 1
            touches_bot = (y + ch) >= (h - 1)
            touches_tb = touches_top and touches_bot
            fill_ratio = float(area) / max(1.0, float(cw * ch))

            is_border_line = cw > 0.45 * w or ch > 0.94 * h or touches_tb
            # Sisa bingkai vertikal / sudut L di tepi kiri-kanan (sering terbaca I, L, 1, Q)
            is_edge_sliver = touches_lr and (
                cw < 0.28 * ref_ch
                or ch < 0.55 * ref_ch
                or (touches_top or touches_bot) and cw < 0.50 * ref_ch
            )
            is_noise = area < min_area
            is_tax_row = (ch < 0.44 * ref_ch) or (y > 0.68 * h and ch < 0.68 * ref_ch)
            is_glare_blob = (cw > 1.45 * ref_ch and fill_ratio > 0.78) or (cw > 0.30 * w and ch < 0.35 * h)

            if is_border_line or is_edge_sliver or is_noise or is_tax_row or is_glare_blob:
                cleaned[labels == i] = 0
                removed += 1

        # Pengaman: jika pembersihan terlalu agresif hingga hampir kosong,
        # kembalikan versi longgar yang hanya membuang noise mikro & garis tepi.
        if np.count_nonzero(cleaned) < cleaned.size * 0.02:
            fallback = morphed_bin.copy()
            for i in range(1, n):
                x, y, cw, ch, area = stats[i]
                if area < min_area or cw > 0.55 * w or ch > 0.94 * h:
                    fallback[labels == i] = 0
            return self._pad(fallback, value=0), removed

        return self._pad(cleaned, value=0), removed

    def _masked_gray_ocr(self, main_gray, roi_cleaned_padded):
        """
        Buat varian grayscale anti-aliased (teks gelap di atas latar putih 255)
        dengan latar belakang di luar masker karakter bersih dijadikan putih murni,
        sehingga pantulan cahaya, baut, dan bingkai hilang tanpa merusak kontur huruf.
        """
        # Ambil kembali masker tanpa padding (14px atas/bawah, 18px kiri/kanan)
        mask = roi_cleaned_padded[14:-14, 18:-18]
        if mask.shape != main_gray.shape:
            mask = cv2.resize(mask, (main_gray.shape[1], main_gray.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Dilasi masker agar tepi anti-aliased huruf tetap utuh
        k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        soft_mask = cv2.dilate(mask, k_dil)

        p_low = float(np.percentile(main_gray, 15))
        p_high = float(np.percentile(main_gray, 99))
        if p_high - p_low > 15.0:
            stretched = np.clip((main_gray.astype(np.float32) - p_low) * (255.0 / (p_high - p_low)), 0, 255).astype(np.uint8)
        else:
            stretched = cv2.normalize(main_gray, None, 0, 255, cv2.NORM_MINMAX)

        inv = cv2.bitwise_not(stretched)
        # Latar di luar masker huruf -> putih bersih 255
        out = np.full_like(inv, 255)
        out[soft_mask > 0] = inv[soft_mask > 0]
        return self._pad(out, value=255)

    def _to_ocr_dark_on_white(self, gray_dark_bg, use_clahe=False):
        """
        Konversi ROI grayscale (latar gelap, teks terang) menjadi teks gelap
        di atas latar putih bersih (mean > 165) dengan padding putih (255),
        agar EasyOCR menerima polaritas optimal tanpa gagal inversi saat ada glare.
        """
        work = gray_dark_bg.copy()
        if use_clahe:
            work = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(work)

        # Normalisasi persentil agar latar benar-benar gelap sebelum di-invert
        p_low = float(np.percentile(work, 15))
        p_high = float(np.percentile(work, 99))
        if p_high - p_low > 15.0:
            stretched = np.clip((work.astype(np.float32) - p_low) * (255.0 / (p_high - p_low)), 0, 255).astype(np.uint8)
        else:
            stretched = cv2.normalize(work, None, 0, 255, cv2.NORM_MINMAX)

        # Invert -> teks gelap (0..60) di atas latar putih (220..255)
        inv = cv2.bitwise_not(stretched)
        # Pastikan mean >= 150 agar recognizer._prepare_for_ocr tidak membalik lagi
        if float(np.mean(inv)) < 145.0:
            inv = cv2.convertScaleAbs(inv, alpha=1.15, beta=30)
        return self._pad(inv, value=255)

    def _extract_tax_roi(self, sharpened_gray):
        """
        Ekstrak ROI Angka Pajak MM.YY di bagian bawah pelat (Tahap 9b):
        y: 58%-98% tinggi, x: 6%-96% lebar agar tidak membawa teks dudukan pelat
        namun tetap menangkap posisi pajak di tengah maupun kanan-bawah pada plat motor/lama.
        """
        h, w = sharpened_gray.shape[:2]
        y1, y2 = int(h * 0.58), int(h * 0.98)
        x1, x2 = int(w * 0.04), w
        crop_gray = sharpened_gray[y1:y2, x1:x2]
        if crop_gray.size == 0:
            crop_gray = sharpened_gray

        tax_gray_raw = cv2.resize(
            crop_gray,
            (max(1, crop_gray.shape[1] * 2), max(1, crop_gray.shape[0] * 2)),
            interpolation=cv2.INTER_CUBIC,
        )
        assisted = self._tophat_assist(tax_gray_raw)
        th, tw = assisted.shape[:2]
        # Hitung ambang Otsu dari 70% sisi kanan-tengah (area angka pajak) agar garis bingkai/logo
        # di pojok kiri bawah tidak menarik ambang Otsu terlalu tinggi.
        otsu_zone = assisted[:, int(tw * 0.28):]
        if otsu_zone.size == 0:
            otsu_zone = assisted
        t_val, _ = cv2.threshold(otsu_zone, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, tax_bin = cv2.threshold(assisted, t_val, 255, cv2.THRESH_BINARY)
        if np.count_nonzero(tax_bin[:, int(tw * 0.28):]) > otsu_zone.size * 0.55:
            tax_bin = cv2.bitwise_not(tax_bin)

        # Bersihkan noise mikro, garis horizontal/bingkai kiri, dan sisa ujung huruf baris utama pada ROI pajak
        n, labels, stats, _ = cv2.connectedComponentsWithStats(tax_bin, connectivity=8)
        min_tax_area = max(10, int(th * tw * 0.002))
        removed_mask = np.zeros((th, tw), dtype=bool)
        for i in range(1, n):
            x, y, cw, ch, area = stats[i]
            touches_border = (y <= 1 and cw > 0.25 * tw) or (y + ch >= th - 1 and cw > 0.25 * tw)
            touches_side_line = (x <= 1 or x + cw >= tw - 1) and (ch > 0.75 * th and cw <= max(4, int(tw * 0.025)))
            touches_top_main = (y <= 1 and (y + ch) < 0.28 * th)
            is_horiz_line = (cw >= max(14, int(tw * 0.09)) and cw >= 2.5 * max(1, ch))
            is_far_left_noise = ((x + cw) <= int(tw * 0.28))
            if (
                area < min_tax_area
                or cw > 0.55 * tw
                or touches_border
                or touches_side_line
                or touches_top_main
                or is_horiz_line
                or is_far_left_noise
            ):
                tax_bin[labels == i] = 0
                if is_horiz_line or is_far_left_noise or touches_border or touches_side_line:
                    removed_mask |= (labels == i)

        # Tutup celah 1-2 piksel pada angka pajak (mis. angka '0' yang agak putus seperti 'G'/'C')
        tax_bin = cv2.morphologyEx(
            tax_bin,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
        )

        tax_gray_clean = tax_gray_raw.copy()
        if removed_mask.any():
            bg_med = int(np.median(tax_gray_clean[~removed_mask])) if (~removed_mask).any() else 0
            tax_gray_clean[removed_mask] = bg_med
        tax_gray_ocr = self._to_ocr_dark_on_white(tax_gray_clean, use_clahe=False)
        return self._pad(tax_bin, value=0), tax_gray_ocr

    def separate_main_and_tax_roi(self, morphed_bin, sharpened_gray):
        """Tahap 9: Pisahkan ROI Nomor Utama dan ROI Nomor Pajak."""
        roi_cleaned, removed_blobs = self._clean_main_components(morphed_bin)
        tax_roi_binary, tax_roi_gray = self._extract_tax_roi(sharpened_gray)
        return roi_cleaned, removed_blobs, tax_roi_binary, tax_roi_gray

    @staticmethod
    def _pad(img, value=0):
        return cv2.copyMakeBorder(img, 14, 14, 18, 18, cv2.BORDER_CONSTANT, value=value)

    def process(self, sharpened_gray):
        """
        Alur lengkap Tahap 7-9:
        Crop ROI Utama (1%-75% tinggi, lebar penuh) -> Thresholding -> Morphological Closing -> Dual-ROI (Utama & Pajak).
        """
        if sharpened_gray is None or sharpened_gray.size == 0:
            raise ValueError("Citra sharpened grayscale kosong!")

        h, w = sharpened_gray.shape[:2]
        # Detector sudah memangkas bingkai luar, jadi jangan memotong tepi kiri/kanan terlalu agresif
        main_gray = sharpened_gray[int(h * 0.01):int(h * 0.72), 0:w]
        if main_gray.size == 0:
            main_gray = sharpened_gray

        primary_mode = "adaptive" if self.method == "adaptive" else "otsu"
        secondary_mode = "otsu" if primary_mode == "adaptive" else "adaptive"

        binary_raw = self._threshold(main_gray, primary_mode)
        morphed = self._close_gaps(binary_raw)
        roi_cleaned, removed_blobs = self._clean_main_components(morphed)
        tax_roi_binary, tax_roi_gray = self._extract_tax_roi(sharpened_gray)

        # Kandidat sekunder: Biner Alternatif (Adaptive) + 2 Varian Grayscale Dark-on-White
        sec_raw = self._threshold(main_gray, secondary_mode)
        sec_morphed = self._close_gaps(sec_raw)
        sec_roi, _ = self._clean_main_components(sec_morphed)

        gray_clean_ocr = self._to_ocr_dark_on_white(main_gray, use_clahe=False)
        gray_masked_otsu = self._masked_gray_ocr(main_gray, roi_cleaned)

        white_ratio = round(float(np.count_nonzero(roi_cleaned) / max(1, roi_cleaned.size)), 3)

        return {
            "upscaled_gray": gray_clean_ocr,
            "binary_raw": binary_raw,
            "morphed": morphed,
            "binary_normalized": roi_cleaned,
            "roi_cleaned": roi_cleaned,
            "removed_blobs": removed_blobs,
            "white_ratio": white_ratio,
            "tax_roi_binary": tax_roi_binary,
            "tax_roi_gray": tax_roi_gray,
            "variants": [sec_roi, gray_clean_ocr, gray_masked_otsu],
        }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Penggunaan: python Script/binarizer.py <path_gambar_plat>")
        sys.exit(0)
    from Script.preprocessor import PlatePreprocessor
    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Gagal membaca gambar: {sys.argv[1]}")
        sys.exit(1)
    pre = PlatePreprocessor().process(img)
    res = PlateBinarizer().process(pre["sharpened"])
    print(f"Binarisasi sukses | roi={res['roi_cleaned'].shape} | removed={res['removed_blobs']} | white={res['white_ratio']}")
