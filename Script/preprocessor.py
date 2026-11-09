import cv2
import numpy as np

class PlatePreprocessor:
    """
    Modul Preprocessing Citra Plat Nomor.
    Menerapkan:
    - Materi P4: Konversi Ruang Warna (BGR ke Grayscale)
    - Materi P6: Histogram Processing (Analisis Statistik Histogram & Smart Conditional CLAHE)
    - Materi P7: Filter Spasial Smoothing (Bilateral Filter untuk reduksi noise sambil menjaga ketajaman tepi karakter)
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8), std_threshold=45.0):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.std_threshold = std_threshold
        self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

    def to_grayscale(self, image_bgr):
        """
        Materi P4: Konversi Ruang Warna.
        Mengubah citra BGR menjadi Grayscale.
        """
        if len(image_bgr.shape) == 2:
            return image_bgr
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

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
        Jika kontras sudah tinggi (std >= threshold) dan pencahayaan tidak ekstrem, CLAHE tidak diperlukan.
        """
        mean_val = float(np.mean(gray_img))
        std_val = float(np.std(gray_img))
        
        # Kondisi butuh CLAHE:
        # 1. Sebaran kontras sempit/pudar (std_val < std_threshold, misal < 45)
        # 2. Atau gambar terlalu gelap gulita (mean_val < 45)
        needs_clahe = (std_val < self.std_threshold) or (mean_val < 45.0)
        
        if needs_clahe:
            reason = f"CLAHE Diterapkan (Kontras Rendah: std={std_val:.1f}, mean={mean_val:.1f})"
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

    def process(self, cropped_plate, force_clahe=None):
        """
        Alur lengkap preprocessing pintar berbasis Histogram:
        Cropped BGR -> Grayscale (P4) -> Smoothing (P7) -> Analisis Histogram (P6) -> Smart CLAHE (P6).
        """
        if cropped_plate is None or cropped_plate.size == 0:
            raise ValueError("Citra plat nomor kosong!")

        # 1. Konversi Ruang Warna (P4)
        gray = self.to_grayscale(cropped_plate)

        # 2. Smoothing (P7)
        smoothed = self.apply_smoothing(gray)

        # 3. Analisis Histogram & Smart CLAHE (P6)
        hist_info = self.analyze_histogram(smoothed)
        
        apply_it = hist_info["needs_clahe"] if force_clahe is None else force_clahe
        if apply_it:
            enhanced = self.enhance_contrast(smoothed)
        else:
            # Jika kontras sudah bagus, gunakan hasil smoothed tanpa merusaknya dengan CLAHE
            enhanced = smoothed.copy()

        return {
            "grayscale": gray,
            "smoothed": smoothed,
            "enhanced": enhanced,
            "clahe_applied": apply_it,
            "hist_mean": hist_info["mean"],
            "hist_std": hist_info["std"],
            "clahe_status": hist_info["status_clahe"]
        }

if __name__ == "__main__":
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_img = os.path.join(base_dir, "Dataset", "Raw", "Indonesian License Plate Dataset", "images", "test", "test001.jpg")
    
    if os.path.exists(test_img):
        img = cv2.imread(test_img)
        pre = PlatePreprocessor()
        out = pre.process(img)
        print("Preprocessing sukses!", out["clahe_status"])
    input("\nTekan Enter untuk menutup jendela ini...")
