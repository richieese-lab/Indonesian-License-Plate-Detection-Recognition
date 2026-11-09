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
│   └── Raw/                                       # Target download dataset via Kaggle API (diabaikan git)
│       ├── Indonesian License Plate Dataset/
│       └── Indonesian License Plate Recognition Dataset/
│
├── Script/                                        # Modul inti pemrosesan citra
│   ├── __init__.py
│   ├── download_dataset.py                        # Script download otomatis via Kaggle API (kagglehub)
│   ├── detector.py                                # P21: Deteksi plat & rasio aspek (YOLO)
│   ├── preprocessor.py                            # P4: Grayscale & P6: CLAHE
│   ├── binarizer.py                               # P12: Binarisasi (Adaptive/Otsu Threshold)
│   ├── recognizer.py                              # EasyOCR & Validasi Pola Plat
│   ├── char_recognizer.py                         # Segmentasi & pengenalan karakter per karakter
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
└── run_on_colab.ipynb                             # Notebook untuk eksekusi lengkap via Colab / Kaggle API
```

---

## 📥 Pengunduhan Dataset (Kaggle API)
Proyek menggunakan dataset publik dari Kaggle: [Indonesian License Plate Dataset](https://www.kaggle.com/datasets/juanthomaswijaya/indonesian-license-plate-dataset).

Untuk mengunduh dataset secara otomatis ke folder lokal `Dataset/Raw/`:
```bash
python Script/download_dataset.py
```
*Catatan: Pastikan token API Kaggle (`kaggle.json`) sudah ada di `~/.kaggle/` atau telah diset via environment variable.*

---

## 🚀 Cara Menjalankan

### Opsi 1: Menjalankan di Google Colab (GPU A100 / T4)
1. Buka Google Colab dan upload notebook **`run_on_colab.ipynb`**.
2. Pilih **Runtime -> Change runtime type -> T4 / A100 GPU**.
3. Jalankan sel secara berurutan:
   * **Tahap 1**: Setup environment & pustaka (`kagglehub`, `ultralytics`, `streamlit`).
   * **Tahap 2**: Download dataset otomatis dari Kaggle API & train YOLOv8.
   * **Tahap 3 - 6**: Uji coba pipeline pengolahan citra & OCR.
   * **Tahap 7**: Jalankan Streamlit Web App dengan public URL via LocalTunnel.

### Opsi 2: Menjalankan di Komputer Lokal
1. Masuk ke direktori folder proyek di terminal / Command Prompt:
   ```bash
   cd "path/ke/folder/proyek/Indonesian-License-Plate-Detection-Recognition"
   ```
2. Pasang dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Unduh dataset (jika ingin melakukan training / evaluasi ground truth lokal):
   ```bash
   python Script/download_dataset.py
   ```
4. Pastikan file bobot `best.pt` sudah tersedia di dalam folder `models/` (bisa diunduh dari hasil training Colab).
5. Jalankan aplikasi Streamlit:
   ```bash
   streamlit run app.py
   ```
6. Buka browser pada alamat `http://localhost:8501`.
