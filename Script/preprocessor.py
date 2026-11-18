import sys
import cv2
import numpy as np


class PlatePreprocessor:
    """
    Modul Preprocessing Citra Plat Nomor (Tahap 4-6).
    Input: `polarity_std` (BGR/Gray dengan standar latar gelap & teks terang).
    Urutan: Grayscale -> Upscale (min 160px) -> Bilateral Smoothing
            -> Conditional CLAHE -> Unsharp Masking + Laplacian Sharpness.
    """

    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8), std_threshold=45.0):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.std_threshold = std_threshold
        self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

    def to_grayscale(self, image_bgr):
        """Konversi citra BGR ke Grayscale (Tahap 4)."""
        if image_bgr.ndim == 2:
            return image_bgr.copy()
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    def upscale_if_small(self, gray_img, target_height=160):
        """Perbesar citra beresolusi rendah dengan interpolasi bicubic (Tahap 4)."""
        h, w = gray_img.shape[:2]
        if 0 < h < target_height:
            scale = target_height / float(h)
            return cv2.resize(
                gray_img,
                (max(1, int(w * scale)), target_height),
                interpolation=cv2.INTER_CUBIC,
            )
        return gray_img

    def apply_smoothing(self, gray_img):
        """Reduksi noise dengan Bilateral Filter tanpa mengaburkan tepi huruf (Tahap 4)."""
        return cv2.bilateralFilter(gray_img, d=5, sigmaColor=35, sigmaSpace=35)

    def analyze_histogram(self, gray_img):
        """Analisis mean & std histogram untuk menentukan perlunya CLAHE (Tahap 5)."""
        mean_val = float(np.mean(gray_img))
        std_val = float(np.std(gray_img))
        needs_clahe = (std_val < self.std_threshold) or (mean_val < 45.0)

        if needs_clahe:
            reason = f"CLAHE Aktif (std={std_val:.1f}, mean={mean_val:.1f})"
        else:
            reason = f"CLAHE Dilewati (std={std_val:.1f}, mean={mean_val:.1f})"

        return {
            "needs_clahe": needs_clahe,
            "mean": round(mean_val, 1),
            "std": round(std_val, 1),
            "status_clahe": reason,
        }

    def enhance_contrast(self, gray_img):
        """Terapkan CLAHE untuk pemerataan kontras adaptif (Tahap 5)."""
        return self.clahe.apply(gray_img)

    def sharpen(self, gray_img):
        """Pertajam tepi karakter dengan Unsharp Masking & ukur ketajaman Laplacian (Tahap 6)."""
        blur = cv2.GaussianBlur(gray_img, (0, 0), 2.0)
        sharpened = cv2.addWeighted(gray_img, 1.6, blur, -0.6, 0)
        sharpness_score = round(float(cv2.Laplacian(sharpened, cv2.CV_64F).var()), 1)
        return sharpened, sharpness_score

    def process(self, plate_img, force_clahe=None):
        """
        Alur lengkap Tahap 4-6:
        Grayscale -> Upscale -> Bilateral -> Conditional CLAHE -> Unsharp Masking.
        """
        if plate_img is None or plate_img.size == 0:
            raise ValueError("Citra plat nomor kosong!")

        gray = self.to_grayscale(plate_img)
        upscaled = self.upscale_if_small(gray, target_height=160)
        smoothed = self.apply_smoothing(upscaled)

        hist_info = self.analyze_histogram(smoothed)
        apply_it = hist_info["needs_clahe"] if force_clahe is None else force_clahe
        enhanced = self.enhance_contrast(smoothed) if apply_it else smoothed.copy()

        sharpened, sharpness_score = self.sharpen(enhanced)

        return {
            "grayscale": gray,
            "smoothed": smoothed,
            "enhanced": enhanced,
            "sharpened": sharpened,
            "sharpness_score": sharpness_score,
            "clahe_applied": apply_it,
            "hist_mean": hist_info["mean"],
            "hist_std": hist_info["std"],
            "clahe_status": hist_info["status_clahe"],
        }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Penggunaan: python Script/preprocessor.py <path_gambar_plat>")
        sys.exit(0)
    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Gagal membaca gambar: {sys.argv[1]}")
        sys.exit(1)
    out = PlatePreprocessor().process(img)
    print(f"Preprocessing sukses | {out['clahe_status']} | Sharpness={out['sharpness_score']}")
