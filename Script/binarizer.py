import sys
import cv2
import numpy as np


class PlateBinarizer:
    """
    Modul Segmentasi Biner, Morfologi & Dual-ROI (Tahap 7-9).
    Input: `sharpened` (Grayscale dengan latar gelap & tulisan terang).
    Standar polaritas biner: Latar Hitam (0) & Karakter Putih (255).
    """

    def __init__(self, method="otsu", block_size=31, c_val=-6):
        self.method = method
        self.block_size = block_size
        self.c_val = c_val

    def _threshold(self, gray_img, mode):
        """Tahap 7: Segmentasi Biner (Otsu / Adaptive) berstandar latar 0 & teks 255."""
        if mode == "adaptive":
            b = self.block_size if self.block_size % 2 else self.block_size + 1
            binary = cv2.adaptiveThreshold(
                gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, b, self.c_val,
            )
        else:
            _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Pastikan piksel latar (mayoritas) bernilai 0 dan karakter bernilai 255
        if np.count_nonzero(binary) > binary.size * 0.55:
            binary = cv2.bitwise_not(binary)
        return binary

    def _close_gaps(self, binary_raw):
        """Tahap 8: Morphological Closing pada karakter putih (255) untuk menyambung celah."""
        h = binary_raw.shape[0]
        k = max(3, (h // 32) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.morphologyEx(binary_raw, cv2.MORPH_CLOSE, kernel)

    def _clean_main_components(self, morphed_bin):
        """
        Ekstrak ROI Nomor Utama (Tahap 9a): hapus bingkai luar, bintik noise,
        dan angka pajak kecil di baris bawah menggunakan Connected Components.
        """
        cleaned = morphed_bin.copy()
        h, w = cleaned.shape[:2]
        n, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        if n <= 1:
            return self._pad(cleaned, value=0), 0

        min_area = max(18, int(h * w * 0.0012))
        candidates = [
            stats[i][3] for i in range(1, n)
            if stats[i][4] >= min_area and stats[i][2] <= 0.5 * w and stats[i][1] < 0.72 * h
        ]
        max_ch = max(candidates) if candidates else int(h * 0.55)

        removed = 0
        for i in range(1, n):
            x, y, cw, ch, area = stats[i]
            is_border_line = cw > 0.55 * w or ch > 0.90 * h
            is_noise = area < min_area
            is_tax_row = (ch < 0.48 * max_ch) or (y > 0.70 * h and ch < 0.65 * max_ch)
            if is_border_line or is_noise or is_tax_row:
                cleaned[labels == i] = 0
                removed += 1

        return self._pad(cleaned, value=0), removed

    def _extract_tax_roi(self, sharpened_gray):
        """
        Ekstrak ROI Angka Pajak MM.YY di bagian tengah-bawah pelat (Tahap 9b):
        y: 62%-93% tinggi, x: 22%-78% lebar agar tidak membawa teks dudukan pelat.
        """
        h, w = sharpened_gray.shape[:2]
        y1, y2 = int(h * 0.62), int(h * 0.93)
        x1, x2 = int(w * 0.22), int(w * 0.78)
        crop_gray = sharpened_gray[y1:y2, x1:x2]
        if crop_gray.size == 0:
            crop_gray = sharpened_gray

        tax_gray = cv2.resize(
            crop_gray,
            (max(1, crop_gray.shape[1] * 2), max(1, crop_gray.shape[0] * 2)),
            interpolation=cv2.INTER_CUBIC,
        )
        _, tax_bin = cv2.threshold(tax_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.count_nonzero(tax_bin) > tax_bin.size * 0.55:
            tax_bin = cv2.bitwise_not(tax_bin)

        # Bersihkan noise mikro pada ROI pajak
        th, tw = tax_bin.shape[:2]
        n, labels, stats, _ = cv2.connectedComponentsWithStats(tax_bin, connectivity=8)
        min_tax_area = max(10, int(th * tw * 0.002))
        for i in range(1, n):
            if stats[i][4] < min_tax_area or stats[i][2] > 0.65 * tw:
                tax_bin[labels == i] = 0

        return self._pad(tax_bin, value=0), self._pad(tax_gray, value=0)

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
        Crop ROI Utama (4%-74% tinggi) -> Thresholding -> Morphological Closing -> Dual-ROI (Utama & Pajak).
        """
        if sharpened_gray is None or sharpened_gray.size == 0:
            raise ValueError("Citra sharpened grayscale kosong!")

        h, w = sharpened_gray.shape[:2]
        main_gray = sharpened_gray[int(h * 0.04):int(h * 0.74), int(w * 0.03):int(w * 0.97)]
        if main_gray.size == 0:
            main_gray = sharpened_gray

        primary_mode = "adaptive" if self.method == "adaptive" else "otsu"
        secondary_mode = "otsu" if primary_mode == "adaptive" else "adaptive"

        binary_raw = self._threshold(main_gray, primary_mode)
        morphed = self._close_gaps(binary_raw)
        roi_cleaned, removed_blobs = self._clean_main_components(morphed)
        tax_roi_binary, tax_roi_gray = self._extract_tax_roi(sharpened_gray)

        # Kandidat sekunder (Adaptive ROI & StretchCLAHE) untuk voting OCR akurat
        sec_raw = self._threshold(main_gray, secondary_mode)
        sec_morphed = self._close_gaps(sec_raw)
        sec_roi, _ = self._clean_main_components(sec_morphed)

        stretched = cv2.normalize(main_gray, None, 0, 255, cv2.NORM_MINMAX)
        stretch_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(stretched)
        stretch_padded = cv2.copyMakeBorder(stretch_clahe, 14, 14, 18, 18, cv2.BORDER_REPLICATE)
        sharp_padded = cv2.copyMakeBorder(main_gray, 14, 14, 18, 18, cv2.BORDER_REPLICATE)

        white_ratio = round(float(np.count_nonzero(roi_cleaned) / max(1, roi_cleaned.size)), 3)

        return {
            "upscaled_gray": sharp_padded,
            "binary_raw": binary_raw,
            "morphed": morphed,
            "binary_normalized": roi_cleaned,
            "roi_cleaned": roi_cleaned,
            "removed_blobs": removed_blobs,
            "white_ratio": white_ratio,
            "tax_roi_binary": tax_roi_binary,
            "tax_roi_gray": tax_roi_gray,
            "variants": [sec_roi, stretch_padded],
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
