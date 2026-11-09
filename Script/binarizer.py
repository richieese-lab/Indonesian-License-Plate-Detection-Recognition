import cv2
import numpy as np

class PlateBinarizer:
    """
    Modul Segmentasi Karakter Plat Nomor.
    Materi yang diterapkan:
    - P8  : Unsharp Masking (mempertajam tepi karakter setelah upscale)
    - P12 : Segmentasi Dasar (Otsu & Adaptive Thresholding)
    - P13-P14 : Morfologi Citra (Closing untuk menyambung karakter yang terputus
                oleh objek tipis seperti kabel/tali, serta pembersihan komponen noise)

    Menghasilkan beberapa varian citra biner agar OCR dapat melakukan voting.
    """
    def __init__(self, method="otsu", block_size=31, c_val=8):
        self.method = method
        self.block_size = block_size
        self.c_val = c_val

    # ------------------------------------------------------------------
    def extract_main_row_roi(self, gray_img):
        """
        Memotong tepi bingkai luar plat. Bagian bawah hanya dipotong sedikit (8%)
        agar bawah angka/huruf tidak terpotong. Angka pajak kecil dibuang kemudian
        lewat filter tinggi komponen (clean_components).
        """
        h, w = gray_img.shape[:2]
        if h > 25 and w > 60:
            y1 = int(h * 0.03)
            y2 = int(h * 0.92)
            x1 = int(w * 0.02)
            x2 = int(w * 0.98)
            roi = gray_img[y1:y2, x1:x2]
            if roi.size > 0:
                return roi
        return gray_img

    def upscale_if_small(self, gray_img, target_height=140):
        h, w = gray_img.shape[:2]
        if 0 < h < target_height:
            scale = target_height / float(h)
            return cv2.resize(gray_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        return gray_img

    def sharpen(self, gray_img):
        """Materi P8: Unsharp Masking."""
        blur = cv2.GaussianBlur(gray_img, (0, 0), 2.0)
        return cv2.addWeighted(gray_img, 1.6, blur, -0.6, 0)

    # ------------------------------------------------------------------
    def _threshold(self, gray_img, mode):
        if mode == "adaptive":
            b = self.block_size if self.block_size % 2 else self.block_size + 1
            return cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, b, self.c_val)
        _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def _normalize_polarity(self, binary):
        """Pastikan latar putih (255) dan karakter hitam (0)."""
        if np.sum(binary == 255) / float(binary.size) < 0.5:
            return cv2.bitwise_not(binary)
        return binary.copy()

    def _close_gaps(self, norm):
        """Materi P13-P14: Closing pada mask karakter untuk menyambung celah (mis. tali di atas angka)."""
        h = norm.shape[0]
        k = max(3, (h // 28) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        char_mask = cv2.bitwise_not(norm)
        closed = cv2.morphologyEx(char_mask, cv2.MORPH_CLOSE, kernel)
        return cv2.bitwise_not(closed)

    def clean_components(self, norm):
        """
        Hapus komponen yang bukan karakter utama:
        - terlalu lebar (garis bingkai), terlalu kecil (bintik/baut),
        - terlalu pendek dibanding karakter tertinggi (angka pajak kecil di bawah plat).
        """
        cleaned = norm.copy()
        h, w = cleaned.shape[:2]
        char_mask = cv2.bitwise_not(cleaned)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(char_mask, connectivity=8)
        if n <= 1:
            return cleaned

        min_area = max(20, int(h * w * 0.0015))
        candidates = [stats[i][3] for i in range(1, n)
                      if stats[i][4] >= min_area and stats[i][2] <= 0.5 * w]
        max_ch = max(candidates) if candidates else h

        for i in range(1, n):
            x, y, cw, ch, area = stats[i]
            if cw > 0.5 * w or area < min_area or ch < 0.5 * max_ch:
                cleaned[labels == i] = 255
        return cleaned

    def _pad(self, img, value=255):
        return cv2.copyMakeBorder(img, 15, 15, 20, 20, cv2.BORDER_CONSTANT, value=value)

    def _build_variant(self, gray, mode):
        b = self._threshold(gray, mode)
        b = self._normalize_polarity(b)
        b = self._close_gaps(b)
        b = self.clean_components(b)
        return self._pad(b)

    # ------------------------------------------------------------------
    def process(self, enhanced_gray):
        roi = self.extract_main_row_roi(enhanced_gray)
        upscaled = self.upscale_if_small(roi)
        sharp = self.sharpen(upscaled)

        primary_mode = "adaptive" if self.method == "adaptive" else "otsu"
        secondary_mode = "otsu" if primary_mode == "adaptive" else "adaptive"

        binary_primary = self._build_variant(sharp, primary_mode)
        binary_secondary = self._build_variant(sharp, secondary_mode)

        sharp_padded = cv2.copyMakeBorder(sharp, 15, 15, 20, 20, cv2.BORDER_REPLICATE)

        return {
            "upscaled_gray": sharp_padded,
            "binary_raw": binary_primary,
            "binary_normalized": binary_primary,
            "variants": [binary_secondary, sharp_padded]
        }

if __name__ == "__main__":
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_img = os.path.join(base_dir, "Dataset", "Raw", "Indonesian License Plate Dataset", "images", "test", "test001.jpg")
    if os.path.exists(test_img):
        img = cv2.imread(test_img, cv2.IMREAD_GRAYSCALE)
        res = PlateBinarizer().process(img)
        print("Binarisasi sukses! Ukuran biner:", res["binary_normalized"].shape)
    input("\nTekan Enter untuk menutup jendela ini...")
