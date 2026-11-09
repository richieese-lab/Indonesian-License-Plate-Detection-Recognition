# Indonesian License Plate Detection & Recognition (ALPR)
Proyek Akhir Mata Kuliah: **Pengolahan Citra Digital dan Visi Komputer**

Sistem ini melakukan lokalisasi pelat nomor kendaraan bermotor Indonesia, peningkatan kualitas citra plat, segmentasi biner, pembacaan teks plat nomor (OCR), serta deteksi otomatis plat yang terhalang/tidak utuh (*partial occlusion*).

---

## 📚 Penerapan Materi Kuliah Slide
Proyek ini mengintegrasikan minimal 3 materi kuliah sesuai ketentuan:
1. **P21. YOLO (Object Detection):** Melokalisasi posisi pelat nomor kendaraan dari citra utuh ke dalam koordinat bounding box dan memotong area ROI plat.
2. **P4. Konversi Ruang Warna:** Mengubah citra plat hasil crop dari BGR menjadi Grayscale.
3. **P6. Histogram Processing (CLAHE):** *Contrast Limited Adaptive Histogram Equalization* untuk meratakan dan mempertajam kontras karakter plat terhadap latar belakangnya.
4. **P12. Segmentasi Dasar:** Menerapkan *Adaptive Gaussian Thresholding* dan *Otsu Thresholding* untuk binarisasi karakter sebelum pembacaan OCR.
5. **Ekstraksi Fitur & Rasio Aspek (P17) / Validasi Oklusi:** Mendeteksi plat nomor yang terhalang/terpotong berdasarkan rasio aspek dan pola sintaks plat nomor Indonesia.

---

## 📁 Struktur Direktori
```text
Indonesian License Plate Detection/
│
├── Dataset/
│   ├── data.yaml                                  # Konfigurasi data training YOLO
│   └── Raw/                                       # Dataset asli dari Kaggle
│       ├── Indonesian License Plate Dataset/
│       └── Indonesian License Plate Recognition Dataset/
│
├── Script/                                        # Modul inti pemrosesan citra
│   ├── __init__.py
│   ├── detector.py                                # P21: Deteksi plat & rasio aspek (YOLO)
│   ├── preprocessor.py                            # P4: Grayscale & P6: CLAHE
│   ├── binarizer.py                               # P12: Binarisasi (Adaptive/Otsu Threshold)
│   ├── recognizer.py                              # EasyOCR & Validasi Pola Plat
│   └── utils.py                                   # Visualisasi garis penunjuk & Ground Truth
│
├── Output/                                        # Folder ekspor hasil proses bertahap
│   ├── 01_cropped_plates/
│   ├── 02_preprocessed/
│   ├── 03_binarized/
│   └── 04_final_detections/
│
├── models/                                        # Tempat bobot model (best.pt)
├── app.py                                         # Aplikasi antarmuka Web berbasis Streamlit
├── requirements.txt                               # Dependensi pustaka Python
└── run_on_colab.ipynb                             # Notebook untuk eksekusi lengkap via Google Drive di Colab
```

---

## 🚀 Cara Menjalankan

### Opsi 1: Menjalankan di Google Colab (GPU A100 / T4)
1. Upload folder proyek `Indonesian License Plate Detection` ke **Google Drive** Anda (di `MyDrive`).
2. Buka Google Colab: upload file **`run_on_colab.ipynb`**.
3. Pilih **Runtime -> Change runtime type -> A100 GPU**.
4. Jalankan sel secara berurutan:
   * Sel 1: Mount Google Drive
   * Sel 2: Install dependencies
   * Sel 3: Training YOLOv8 (hanya butuh ~2-3 menit di A100 GPU)
   * Sel 4: Jalankan Streamlit Web App dengan public URL via LocalTunnel.

### Opsi 2: Menjalankan di Komputer Lokal
1. Buka Terminal / PowerShell di folder proyek ini:
   ```bash
   cd "D:\Richard\Semester 5\Pengolahan Citra Digital dan Visi Komputer\Project\Indonesian License Plate Detection"
   ```
2. Pastikan dependencies sudah terpasang:
   ```bash
   pip install -r requirements.txt
   ```
3. Pastikan file bobot `best.pt` sudah ada di folder `models/` (bisa diunduh dari hasil training Colab).
4. Jalankan aplikasi Streamlit:
   ```bash
   streamlit run app.py
   ```
5. Buka browser pada alamat `http://localhost:8501`.
