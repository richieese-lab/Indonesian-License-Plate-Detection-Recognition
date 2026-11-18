import os
import glob
import cv2
import numpy as np


class PlatePreprocessor:
    """
    Modul Preprocessing Citra Plat Nomor (Tahap 4 - 6).
    Input: Citra pelat `polarity_std` yang SUDAH berstandar Background Hitam (Gelap) & Tulisan Putih (Terang).
    Urutan proses:
    1. Materi P4: Konversi Ruang Warna (BGR ke Grayscale) + Upscaling (INTER_CUBIC)
    2. Materi P7: Filter Spasial Smoothing (Bilateral Filter untuk reduksi noise sambil menjaga tepi karakter)
    3. Materi P6: Histogram Processing (Analisis Statistik Histogram & Smart Conditional CLAHE)
    4. Materi P8: Penajaman Citra (Unsharp Masking) + Perhitungan Skor Ketajaman (Laplacian Variance)
    """

    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8), std_threshold=45.0, target_height=160):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.std_threshold = std_threshold
        self.target_height = target_height
        self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

    def to_grayscale(self, image_bgr):
        """
        Materi P4: Konversi Ruang Warna.
        Mengubah citra BGR menjadi Grayscale (1 channel).
        """
        if len(image_bgr.shape) == 2:
            return image_bgr.copy()
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    def upscale_if_small(self, gray_img, target_height=None):
        """
        Memperbesar citra pelat menggunakan interpolasi Bicubic jika tingginya < target_height (default 160 px),
        agar detail karakter utama dan angka pajak kecil tetap tajam saat diproses.
        """
        if target_height is None:
            target_height = self.target_height
        h, w = gray_img.shape[:2]
        if 0 < h < target_height:
            scale = float(target_height) / float(h)
            return cv2.resize(gray_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        return gray_img.copy()

    def apply_smoothing(self, gray_img):
        """
        Materi P7: Filter Domain Spasial (Smoothing).
        Menggunakan Bilateral Filter untuk menghilangkan noise/bintik debu
        tanpa mengaburkan tepi (edge) huruf dan angka pada plat.
        """
        return cv2.bilateralFilter(gray_img, d=5, sigmaColor=35, sigmaSpace=35)

    def analyze_histogram(self, gray_img):
        """
        Materi P6: Analisis Statistik Histogram Citra.
        Menghitung rata-rata intensitas (mean) dan sebaran kontras (standard deviation) dari histogram.
        Jika kontras sudah tinggi (std >= threshold) dan pencahayaan tidak terlalu gelap, CLAHE dilewati.
        """
        mean_val = float(np.mean(gray_img))
        std_val = float(np.std(gray_img))

        needs_clahe = (std_val < self.std_threshold) or (mean_val < 45.0)

        if needs_clahe:
            reason = f"CLAHE Diterapkan (Kontras Rendah/Gelap: std={std_val:.1f}, mean={mean_val:.1f})"
        else:
            reason = f"CLAHE Dilewati - Kontras Sudah Baik (std={std_val:.1f}, mean={mean_val:.1f})"

        return {
            "needs_clahe": needs_clahe,
            "mean": round(mean_val, 1),
            "std": round(std_val, 1),
            "status_clahe": reason
        }

    def enhance_contrast(self, gray_img):
        """
        Materi P6: Histogram Processing (CLAHE).
        """
        return self.clahe.apply(gray_img)

    def sharpen(self, gray_img):
        """
        Materi P8: Penajaman Citra (Unsharp Masking).
        Mengurangi citra dengan hasil Gaussian Blur-nya secara terbobot untuk mempertegas tepi karakter,
        serta menghitung skor ketajaman menggunakan Variance of Laplacian.
        """
        blur = cv2.GaussianBlur(gray_img, (0, 0), 2.0)
        sharpened = cv2.addWeighted(gray_img, 1.6, blur, -0.6, 0)
        sharpness_score = round(float(cv2.Laplacian(sharpened, cv2.CV_64F).var()), 1)
        return sharpened, sharpness_score

    def process(self, plate_img, force_clahe=None):
        """
        Alur lengkap preprocessing sesuai diagram baru:
        to_grayscale -> upscale_if_small -> apply_smoothing -> analyze_histogram & enhance_contrast (CLAHE) -> sharpen.
        """
        if plate_img is None or plate_img.size == 0:
            raise ValueError("Citra plat nomor kosong!")

        # 1. Konversi Ruang Warna (P4) & Upscaling
        gray_raw = self.to_grayscale(plate_img)
        gray = self.upscale_if_small(gray_raw, target_height=self.target_height)

        # 2. Smoothing Bilateral Filter (P7)
        smoothed = self.apply_smoothing(gray)

        # 3. Analisis Histogram & Conditional CLAHE (P6)
        hist_info = self.analyze_histogram(smoothed)
        apply_it = hist_info["needs_clahe"] if force_clahe is None else bool(force_clahe)
        if apply_it:
            enhanced = self.enhance_contrast(smoothed)
        else:
            enhanced = smoothed.copy()

        # 4. Penajaman Citra / Unsharp Masking (P8)
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
            "clahe_status": hist_info["status_clahe"]
        }


def resolve_sample_image(base_dir):
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
        os.path.join(base_dir, "Output", "01_cropped_plates"),
        os.path.join(base_dir, "Dataset", "images", "Indonesian License Plate Dataset", "images", "test"),
        os.path.join(base_dir, "Dataset", "Raw", "Indonesian License Plate Dataset", "images", "test"),
    ]
    for folder in local_candidates:
        if os.path.exists(folder):
            imgs = sorted(glob.glob(os.path.join(folder, "*.jpg")))
            if imgs:
                return imgs[0]
    return None


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample_path = resolve_sample_image(base_dir)

    if sample_path and os.path.exists(sample_path):
        img = cv2.imread(sample_path)
        pre = PlatePreprocessor()
        out = pre.process(img)
        out_dir = os.path.join(base_dir, "Output", "02_preprocessed")
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, "test_smoothed.jpg"), out["smoothed"])
        cv2.imwrite(os.path.join(out_dir, "test_clahe.jpg"), out["enhanced"])
        cv2.imwrite(os.path.join(out_dir, "test_sharpened.jpg"), out["sharpened"])
        print("Preprocessing sukses!")
        print(f" - Status CLAHE     : {out['clahe_status']}")
        print(f" - Skor Ketajaman   : {out['sharpness_score']} (Laplacian Variance)")
        print(f" - Hasil disimpan di: {out_dir}")
    else:
        print("Tidak ditemukan gambar uji di Kaggle API maupun folder lokal.")
