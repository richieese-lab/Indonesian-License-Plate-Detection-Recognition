# ALPR Studio Indonesia — Deteksi Pelat Nomor, Klasifikasi Warna, Custom YOLOv8 OCR & Status Pajak

Proyek Akhir Mata Kuliah **Pengolahan Citra Digital dan Visi Komputer**

Sistem **Automatic License Plate Recognition (ALPR)** Indonesia end-to-end berbasis **YOLOv8s Plate Detector**, **10-Tahap Pengolahan Citra Digital (OpenCV)**, dan **Custom YOLOv8n Character OCR** (36 kelas `0–9`, `A–Z`) dengan mode **Hybrid EasyOCR** untuk pembacaan masa berlaku pajak (`MM.YY`). Dilengkapi akselerasi **GPU CUDA otomatis**, koreksi kemiringan (*deskewing*), pemangkasan bingkai/casing (*margin trimming*), identifikasi golongan warna kendaraan (HSV), serta validasi standar TNKB Indonesia (**Perpol No. 7 Tahun 2021**).

---

## Fitur & Kebaruan Utama (Novelty)

1. **Custom YOLOv8n Character OCR (`models/ocr_best.pt`) & Multi-Input Voting**
   - Menggantikan pembacaan nomor utama dari OCR generik ke model **YOLOv8n Character Detection & Recognition** yang dilatih khusus pada karakter pelat nomor Indonesia (36 kelas: `0–9` dan `A–Z`).
   - Dilengkapi *1D Horizontal Suppression* (mencegah duplikasi huruf pada posisi horizontal yang sama), pengurutan kiri-ke-kanan (`x_center`), dan **Voting Multi-Kandidat** (*Citra Utama Deskewed*, *Grayscale Tajam*, *Biner ROI Utama Otsu*, serta *Varian Biner/Grayscale*).
   - Validasi dan koreksi posisional mengikuti aturan resmi **TNKB Indonesia (Perpol No. 7 Tahun 2021)** dengan daftar 64 kode wilayah resmi (`[Kode Wilayah 1–2 Huruf] [Angka 1–9999 tanpa awalan 0] [1–3 Huruf Seri]`).

2. **Klasifikasi Warna Pelat & Penanganan Casing Hitam (`PlateColorClassifier` & Polaritas Adaptif)**
   - **Hitam**: Kendaraan Pribadi (Format Lama) — polaritas latar gelap & teks terang dipertahankan.
   - **Putih**: Kendaraan Pribadi (Format Baru) — di-invert ke standar latar gelap & teks putih.
   - **Kuning**: Kendaraan Angkutan Umum / Barang — di-invert ke standar latar gelap & teks putih.
   - **Merah**: Kendaraan Dinas Pemerintah — polaritas latar merah gelap & teks putih dipertahankan.
   - **Deteksi Berbasis Pita Tulisan & Peredaman Casing Hitam:** Sampling Otsu difokuskan pada pita baris huruf utama (`y = 18%..58%`, `x = 12%..88%`) dengan verifikasi komponen karakter (`_main_text_box`), serta meredam casing/dudukan plastik hitam di luar area dalam pelat (`plate_interior`) saat inversi agar casing hitam pada pelat putih/kuning tidak berubah menjadi blok putih yang merusak segmentasi biner.

3. **Deskewing Tahan Kontras Ekstrem & Pemangkasan Pinggiran Cerdas (`trim_plate_margins`)**
   - Estimasi sudut kemiringan ($\pm 45^\circ$) menggunakan variansi proyeksi horizontal Otsu dengan **fallback morfologi TopHat + BlackHat** agar tetap akurat meskipun sudut crop kemasukan objek terang (helm) atau latar malam yang gelap.
   - Pemangkasan bingkai luar berbasis *integral proyeksi* dan *morphological opening* yang melindungi batas kotak huruf utama sekaligus batas angka pajak (`_tax_content_bounds` + margin aman) agar digit tahun pajak di pojok kanan-bawah tidak ikut terpotong.

4. **Segmentasi Dual-ROI & Pembacaan Pajak Hybrid (`MM.YY`)**
   - Memisahkan **ROI Nomor Utama** (`roi_cleaned`) dari **ROI Angka Pajak** (`tax_roi_binary` & `tax_roi_gray`).
   - Pada ROI Pajak, ambang Otsu dihitung khusus dari zona tengah-kanan (`28%–100%` lebar) disertai penghapusan komponen garis horizontal (`cw >= 2.5 * ch`) dan noise/logo dealer di sisi kiri bawah.
   - Menggunakan **Mode Hybrid** (`PlateRecognizer.recognize_tax` berbasis EasyOCR untuk teks kecil `MM.YY` dengan *fallback* otomatis ke Custom YOLOv8n OCR `_prepare_tax_crop`), kamus koreksi digit pajak (`O/G/D/Q/C/U -> 0`, `I/L -> 1`, `Z -> 2`, `S -> 5`, `B -> 8`), dan penentuan status **`AKTIF`** vs **`MATI / KEDALUWARSA`**.

5. **Akselerasi GPU Otomatis (`CUDA:0`) & Caching Sesi Streamlit**
   - `PlateDetector` dan `CustomPlateRecognizer` otomatis mendeteksi dan memakai GPU NVIDIA (`torch.cuda.is_available() -> device=0`) dengan fallback otomatis ke CPU.
   - Hasil deteksi dan pemrosesan seluruh pelat dalam satu foto disimpan di `st.session_state` sehingga berpindah pilihan pelat pada dropdown inspeksi berlangsung instan tanpa menjalankan ulang pipeline.

---

## Arsitektur Pipeline (11 Tahap Terintegrasi)

| Tahap | Proses | Modul | Keterangan |
| :---: | :--- | :--- | :--- |
| **1** | Deteksi & Crop ROI YOLO | [`Script/detector.py`](Script/detector.py) | YOLOv8s (`models/best.pt`, `imgsz` adaptif `640`/`960`, *auto-threshold*, fallback rotasi $\pm 35^\circ$, saring deteksi palsu) |
| **2** | Deskewing & Margin Trim | [`Script/detector.py`](Script/detector.py) | Proyeksi horizontal + fallback TopHat/BlackHat, rotasi dengan konteks foto asli, & `trim_plate_margins()` (proteksi teks utama + pajak) |
| **3** | Klasifikasi Warna HSV & Polaritas | [`Script/detector.py`](Script/detector.py) | Identifikasi pelat `Hitam`/`Putih`/`Kuning`/`Merah`, standardisasi latar gelap–tulisan terang, & peredaman casing hitam luar |
| **4** | Grayscale, Glare Norm & Bilateral | [`Script/preprocessor.py`](Script/preprocessor.py) | Konversi Grayscale, peredaman pantulan cahaya lokal, upscale ke tinggi `160px`, & Bilateral Filter (`d=5, sigma=35`) |
| **5** | Conditional CLAHE | [`Script/preprocessor.py`](Script/preprocessor.py) | Peningkatan kontras lokal adaptif (aktif otomatis jika `std < 45` atau `mean < 45`) |
| **6** | Unsharp Masking | [`Script/preprocessor.py`](Script/preprocessor.py) | Penajaman tepi karakter (`GaussianBlur` + `addWeighted`) & perhitungan skor ketajaman Laplacian |
| **7** | Segmentasi Biner | [`Script/binarizer.py`](Script/binarizer.py) | Binerisasi TopHat-assisted Otsu & Adaptive Thresholding (latar `0`, teks `255`) |
| **8** | Perbaikan Morfologi | [`Script/binarizer.py`](Script/binarizer.py) | *Morphological Closing* adaptif untuk menyambung celah goresan huruf tanpa menyatukan karakter bersebelahan |
| **9** | Segmentasi Dual-ROI | [`Script/binarizer.py`](Script/binarizer.py) | Pemisahan **ROI Nomor Utama** (`[09a]`) dan **ROI Pajak `MM.YY`** (`[09b]`) beserta pembersihan garis horizontal & noise tepi |
| **10** | Custom YOLOv8 OCR & Pajak Hybrid | [`Script/custom_ocr.py`](Script/custom_ocr.py), [`Script/recognizer.py`](Script/recognizer.py) | Deteksi karakter 36 kelas (`models/ocr_best.pt`), voting multi-varian, validasi TNKB Perpol No. 7/2021, & pembacaan pajak hybrid |
| **11** | Visualisasi Studio Split-Pane | [`app.py`](app.py), [`Script/utils.py`](Script/utils.py) | Antarmuka Streamlit Dark Precision Studio, KPI strip, replika visual pelat, tabel komparasi kandidat OCR, & grid evaluasi 10 tahap PCD |

---

## Struktur Direktori

```text
Indonesian-License-Plate-Detection-Recognition/
├── .streamlit/
│   └── config.toml                # Konfigurasi tema Dark Precision Streamlit
├── Script/
│   ├── __init__.py
│   ├── detector.py                # Tahap 1-3 : YOLOv8s detector, deskewing, trim, warna HSV & polaritas
│   ├── preprocessor.py            # Tahap 4-6 : Grayscale, glare norm, upscale, Bilateral, CLAHE, Unsharp
│   ├── binarizer.py               # Tahap 7-9 : Otsu/Adaptive, Morph Closing, Dual-ROI Nomor Utama & Pajak
│   ├── custom_ocr.py              # Tahap 10a : Custom YOLOv8n Character OCR (36 kelas), voting & TNKB
│   ├── recognizer.py              # Tahap 10b : Modul EasyOCR (dipakai untuk pembacaan ROI pajak hybrid)
│   ├── plate_tax_pipeline.py      # Pipeline referensi evaluasi batch nomor pelat & pajak
│   └── utils.py                   # Tahap 11  : Anotasi bounding box gambar, badge status & ground truth
├── models/
│   ├── best.pt                    # Bobot YOLOv8s untuk deteksi kotak pelat nomor kendaraan
│   └── ocr_best.pt                # Bobot Custom YOLOv8n untuk deteksi & klasifikasi 36 karakter OCR
├── app.py                         # Aplikasi Web Utama Streamlit (ALPR Studio Indonesia)
├── evaluate_plate_tax.py          # Skrip evaluasi metrik batch
├── MASUKAN_TEKNIS_TIM.md          # Dokumentasi catatan evaluasi & perbaikan teknis tim
├── requirements.txt               # Daftar dependensi Python
└── README.md
```

---

## Ringkasan Performa Model

1. **Model Deteksi Pelat Nomor (`models/best.pt` — YOLOv8s, `test` split 197 pelat):**
   - **Precision:** `93.71%`
   - **Recall:** `98.37%`
   - **mAP@0.5:** `97.87%`
   - **mAP@0.5:0.95:** `65.06%`

2. **Model Custom Character OCR (`models/ocr_best.pt` — YOLOv8n 36 Kelas, `test` split 197 pelat / 1.455 karakter):**
   - **Precision:** `97.70%`
   - **Recall:** `95.50%`
   - **mAP@0.5:** `98.35%`
   - **Character-Level Accuracy (`1 - CER`):** `97.46%`
   - **Exact Plate Match (Raw sebelum koreksi TNKB):** `85.79%` (`169/197` pelat benar utuh)

---

## Cara Menjalankan Project

### 1. Prasyarat & Instalasi Dependensi
Pastikan menggunakan **Python 3.10+** dan pasang seluruh paket yang dibutuhkan dari folder root project:

```powershell
pip install -r requirements.txt
```

> **Opsional (Akselerasi GPU NVIDIA CUDA):**  
> Jika laptop/PC memiliki GPU NVIDIA, pasang PyTorch versi CUDA (misalnya `cu121`) agar `PlateDetector` dan `CustomPlateRecognizer` otomatis berjalan di `CUDA:0`:
> ```powershell
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 2. Pastikan File Bobot Model Tersedia
Pastikan kedua file bobot model berikut ada di dalam folder `models/`:
- `models/best.pt` — Model YOLOv8s pendeteksi lokasi pelat nomor.
- `models/ocr_best.pt` — Model Custom YOLOv8n pembaca karakter `0–9` & `A–Z`.

### 3. Menjalankan Aplikasi Web Streamlit
Jalankan perintah berikut dari terminal di direktori utama project:

```powershell
python -m streamlit run app.py
```

Aplikasi akan terbuka secara otomatis di browser pada alamat **`http://localhost:8501`**.

### 4. Panduan Penggunaan di Antarmuka Streamlit
1. **Pilih Sumber Citra (Sidebar Kiri):**
   - **Upload Gambar:** Unggah foto kendaraan (`.jpg`, `.jpeg`, `.png`, `.webp`) — mendukung foto dengan satu maupun banyak kendaraan sekaligus.
   - **Sampel Cache Kaggle:** Pilih dari sampel dataset uji jika tersedia di cache lokal.
   - **URL Gambar:** Tempel tautan langsung ke gambar kendaraan di internet.
2. **Atur Opsi Inspeksi (Sidebar Kiri):**
   - **Saring Deteksi Palsu:** Menyembunyikan kandidat non-pelat yang tidak lolos validasi format TNKB.
   - **Mode Potongan Pelat Langsung:** Aktifkan jika gambar yang diunggah sudah berupa foto *close-up* potongan pelat nomor.
   - **Tampilkan Grid Evaluasi 10 Tahap PCD:** Menampilkan visualisasi lengkap tahap `[01]` hingga `[09b]` untuk pelat yang sedang dipilih.
3. **Inspeksi Multi-Pelat (Panel Kanan):**
   - Gunakan dropdown **"Pilih Pelat untuk Diinspeksi"** untuk berpindah antar-pelat yang terdeteksi secara instan (tanpa proses *re-run* ulang) dan melihat replika visual pelat, rincian telemetri, tabel voting kandidat OCR, serta hasil tahap pengolahan citra digitalnya.

### 5. Menguji Modul Secara Mandiri lewat Terminal (Opsional)
Setiap modul di dalam folder `Script/` juga dapat diuji langsung melalui terminal:

```powershell
python Script/detector.py path/ke/foto_kendaraan.jpg
python Script/preprocessor.py path/ke/crop_pelat.jpg
python Script/binarizer.py path/ke/crop_pelat.jpg
python Script/recognizer.py
```
