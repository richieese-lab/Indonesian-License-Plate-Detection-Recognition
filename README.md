# Indonesian License Plate Detection & Recognition (ALPR)
Proyek Akhir Mata Kuliah: **Pengolahan Citra Digital dan Visi Komputer**

Sistem ini melakukan lokalisasi pelat nomor kendaraan bermotor Indonesia, koreksi kemiringan (*deskewing*), standarisasi polaritas warna, peningkatan kualitas citra bertahap, segmentasi *Dual-ROI*, pembacaan teks pelat nomor (OCR), deteksi otomatis pelat yang terhalang/tidak utuh (*partial occlusion*), serta dilengkapi **2 Kebaruan (*Novelty*)**:
1. **Identifikasi Warna Pelat & Klasifikasi Jenis Kendaraan (Ruang Warna HSV):** Mengidentifikasi pelat **Hitam & Putih** (Kendaraan Pribadi), **Kuning** (Kendaraan Angkutan Orang/Barang), dan **Merah** (Kendaraan Dinas Pemerintah).
2. **Deteksi Nomor Pajak (`MM.YY`) & Klasifikasi Status Pajak:** Memisahkan ROI baris angka pajak di bagian bawah pelat, mengoreksi distorsi emboss huruf-ke-angka (misal `G1.28` $\rightarrow$ `01.28`), dan menentukan apakah masa pajak kendaraan masih **AKTIF** atau **MATI / KEDALUWARSA**.

---

## 📚 Penerapan Materi Kuliah & Alur Sistem (11 Tahap)
Proyek ini mengintegrasikan materi kuliah secara berurutan sesuai diagram **Alur Sistem & Box Evaluasi**:
1. **P21. YOLOv8n (Object Detection):** Melokalisasi posisi pelat nomor kendaraan ke dalam *bounding box*, menghitung rasio aspek ($W/H$), dan memotong (*crop*) area ROI pelat.
2. **Transformasi Geometris (*Deskewing*):** Mendeteksi sudut kemiringan pelat dan meluruskannya dengan rotasi *affine*.
3. **P4. Ruang Warna (HSV) & Standarisasi Polaritas:** Mengklasifikasikan warna pelat nomor untuk menentukan jenis kendaraan, lalu menstandarkan polaritas citra menjadi **Background Hitam & Tulisan Putih**.
4. **P4 & P7. Grayscale & Bilateral Smoothing:** Mengonversi citra ke *Grayscale*, melakukan *upscaling*, dan mereduksi *noise* bintik menggunakan *Bilateral Filter* tanpa mengaburkan tepi karakter.
5. **P6. Histogram Processing (Conditional CLAHE):** Menganalisis statistik histogram ($\mu$ dan $\sigma$) dan menerapkan CLAHE secara adaptif hanya jika kontras citra rendah atau gelap.
6. **P8. Penajaman Citra (*Unsharp Masking*):** Mempertegas tepi huruf dan angka sebelum masuk ke tahap binarisasi.
7. **P12. Segmentasi Dasar (Thresholding):** Menerapkan *Otsu Thresholding* dan *Adaptive Gaussian Thresholding* untuk menghasilkan citra biner (latar hitam `0`, teks putih `255`).
8. **P13–P14. Perbaikan Morfologi:** Menerapkan *Morphological Closing* pada piksel karakter untuk menyambung huruf/angka yang terputus akibat goresan atau baut.
9. **P12 & P17. Segmentasi Dual-ROI (Pemisahan Nomor Utama & Angka Pajak):** Memisahkan komponen karakter baris utama (`roi_cleaned`) dari sub-area baris masa berlaku pajak (`tax_roi_binary` & `tax_roi_gray`) di bagian tengah-bawah pelat.
10. **Multi-Input OCR, Validasi Regex & Klasifikasi Pajak:** Menjalankan *voting* EasyOCR pada 3 kandidat citra untuk nomor utama serta membaca `MM.YY` pada ROI pajak untuk menentukan status **Pajak Aktif vs Pajak Mati**.
11. **Visualisasi Streamlit & Box Evaluasi 10 Tahap:** Menampilkan gambar akhir beranotasi (*Bounding Box + Leader Line + Badge 2 Baris*), Tabel Komparasi OCR, dan Box Evaluasi seluruh tahapan PCD.

---

## 📁 Struktur Direktori
```text
Indonesian-License-Plate-Detection-Recognition/
│
├── Dataset/
│   ├── data.yaml                                  # Konfigurasi path & kelas untuk training YOLO
│   └── images/                                    # Folder dataset citra plat nomor & karakter
│       ├── Indonesian License Plate Dataset/      # Dataset deteksi plat nomor kendaraan
│       └── Indonesian License Plate Recognition Dataset/ # Dataset pengenalan karakter (0-9, A-Z)
│
├── Script/                                        # Modul inti pemrosesan citra & pipeline ALPR
│   ├── __init__.py                                # Inisialisasi package Python
│   ├── download_dataset.py                        # Utilitas unduh dataset otomatis via Kaggle API
│   ├── detector.py                                # Tahap 1-3: Deteksi YOLOv8, Deskewing, Warna HSV & Polaritas
│   ├── preprocessor.py                            # Tahap 4-6: Grayscale, Smoothing, Conditional CLAHE & Unsharp Masking
│   ├── binarizer.py                               # Tahap 7-9: Segmentasi Biner, Morfologi & Pemisahan Dual-ROI (Utama & Pajak)
│   ├── recognizer.py                              # Tahap 10: Multi-Input OCR Voting, Regex Plat & Klasifikasi Status Pajak
│   ├── char_recognizer.py                         # Alternatif Tahap 10: Pengenalan karakter berbasis YOLO (36 kelas)
│   └── utils.py                                   # Tahap 11: Visualisasi anotasi 2-baris, Ground Truth & helper
│
├── Output/                                        # Folder penyimpanan hasil pemrosesan bertahap
│   ├── 01_cropped_plates/                         # Hasil potongan ROI plat dari YOLO
│   ├── 02_preprocessed/                           # Hasil citra setelah Grayscale, CLAHE & Unsharp Masking
│   ├── 03_binarized/                              # Hasil segmentasi biner & Dual-ROI
│   └── 04_final_detections/                       # Hasil akhir gambar beranotasi lengkap
│
├── models/                                        # Folder bobot model terlatih (best.pt / best_char.pt)
├── app.py                                         # Aplikasi Web interaktif berbasis Streamlit + Box Evaluasi 10 Tahap
├── Dokumen_Rancangan_Sprint_2_Jam.docx            # Dokumen rancangan sistem, kebaruan & pembagian kerja tim
├── requirements.txt                               # Daftar dependensi pustaka Python
├── run_on_colab.ipynb                             # Notebook eksekusi lengkap & training via Google Colab
├── run_pipeline.ipynb                             # Notebook uji coba & demonstrasi pipeline bertahap
└── .gitignore                                     # Aturan file/folder yang diabaikan Git
```

---

## 🧩 Penjelasan Modul Python pada Folder `Script/` (Urutan & Fungsinya)

Seluruh logika pemrosesan citra dipecah menjadi modul-modul terpisah di dalam folder `Script/`. Berikut adalah **urutan eksekusi** beserta fungsi masing-masing file:

### A. Tahap Persiapan Data (Pra-Pipeline)
0. **`Script/download_dataset.py`**
   * **Fungsi:** Mengunduh dan mengekstrak *Indonesian License Plate Dataset* secara otomatis dari Kaggle menggunakan `kagglehub` ke direktori lokal.

### B. Urutan Eksekusi Pipeline Utama (`app.py` / `run_pipeline.ipynb`)
1. **`Script/detector.py` (`PlateDetector`) — Tahap 1 s/d 3: Deteksi, Deskewing, Warna HSV & Standar Polaritas**
   * **Fungsi:** Memuat model YOLOv8 (`models/best.pt`), memotong area pelat (*crop ROI*), menghitung rasio aspek ($W/H$), meluruskan kemiringan pelat (`deskew_plate`), mengklasifikasikan warna pelat & jenis kendaraan di ruang HSV (`classify_plate_color`), serta menyeragamkan polaritas citra menjadi berlatar hitam dan bertuliskan putih (`standardize_polarity`).
2. **`Script/preprocessor.py` (`PlatePreprocessor`) — Tahap 4 s/d 6: Grayscale, Smoothing, Smart CLAHE & Sharpening**
   * **Fungsi:** Mengubah citra pelat ke **Grayscale** (P4), melakukan *upscaling*, mereduksi *noise* menggunakan **Bilateral Filter** (P7), menganalisis histogram ($\mu$ dan $\sigma$) untuk menerapkan **Conditional CLAHE** (P6), serta menajamkan tepi huruf dengan **Unsharp Masking** (P8) beserta perhitungan skor ketajaman *Laplacian variance*.
3. **`Script/binarizer.py` (`PlateBinarizer`) — Tahap 7 s/d 9: Segmentasi Biner, Morfologi & Pemisahan Dual-ROI**
   * **Fungsi:** Melakukan segmentasi biner dengan **Otsu / Adaptive Thresholding** (P12), menyambung huruf yang terputus lewat **Morphological Closing** (P13–P14), serta memisahkan citra menjadi **2 ROI**: `roi_cleaned` (baris nomor utama pelat tanpa gangguan pajak) dan `tax_roi_binary` / `tax_roi_gray` (sub-area angka masa berlaku pajak `MM.YY` di bagian tengah-bawah pelat).
4. **`Script/recognizer.py` (`PlateRecognizer`) — Tahap 10: Pembacaan Teks (OCR), Validasi Regex & Klasifikasi Pajak**
   * **Fungsi:**
     * `recognize()`: Menjalankan **EasyOCR** pada 3 kandidat citra (Biner ROI Otsu, Grayscale Unsharp, dan Biner Adaptive), menghitung rata-rata *confidence*, melakukan *voting*, memvalidasi pola Regex pelat Indonesia, serta menghasilkan tabel komparasi `all_candidates`.
     * `recognize_tax()`: Membaca ROI angka pajak (`MM.YY`), mengoreksi karakter huruf yang mirip angka (`G/O/D -> 0`, `I/L -> 1`, `S -> 5`, `B -> 8`), dan membandingkan bulan/tahun pajak dengan tanggal saat ini untuk menentukan status **AKTIF** atau **MATI / KEDALUWARSA**.
   * **Alternatif Tahap 10 — `Script/char_recognizer.py` (`CharRecognizer`):** Modul opsional berbasis YOLO (36 kelas karakter `0-9` dan `A-Z`).
5. **`Script/utils.py` — Tahap 11: Visualisasi Anotasi, Ground Truth & Utilitas**
   * **Fungsi:**
     * `draw_plate_annotations()`: Menggambar *bounding box* berwarna sesuai status, garis penunjuk siku (*leader line*), dan *badge* 2 baris (Nomor Pelat + Info Warna Pelat & Status Pajak).
     * `get_ground_truth()`: Membaca kunci jawaban dari folder `labelswithLP` untuk evaluasi akurasi.
     * `save_step_output()`: Menyimpan gambar hasil tiap tahapan ke folder `Output/`.

---

## 🚀 Cara Menjalankan di Komputer Lokal
1. Masuk ke direktori proyek di terminal:
   ```bash
   cd "D:\S.Kom Project\Semester 5\Pengolahan Citra Digital dan Visi Komputer\Project Akhir\Indonesian-License-Plate-Detection-Recognition"
   ```
2. Pasang dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan aplikasi Streamlit:
   ```bash
   streamlit run app.py
   ```
4. Buka browser pada alamat `http://localhost:8501`.
