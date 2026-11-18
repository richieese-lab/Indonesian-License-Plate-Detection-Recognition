# ALPR Studio Indonesia — Deteksi Pelat Nomor, Klasifikasi Warna & Status Pajak

Proyek Akhir Mata Kuliah **Pengolahan Citra Digital dan Visi Komputer**

Sistem Automatic License Plate Recognition (ALPR) Indonesia berbasis **YOLOv8s**, **OpenCV Digital Image Processing**, dan **EasyOCR** yang dilengkapi kalibrasi ambang deteksi otomatis, koreksi kemiringan (*deskewing*), identifikasi golongan warna kendaraan, serta pembacaan masa berlaku pajak (`MM.YY`).

---

## Kebaruan Utama (Novelty)

1. **Klasifikasi Warna Pelat & Jenis Kendaraan (Ruang Warna HSV)**
   - **Hitam**: Kendaraan Pribadi (Format Lama) — polaritas latar gelap & teks terang dipertahankan.
   - **Putih**: Kendaraan Pribadi (Format Baru) — di-invert agar berstandar latar gelap & teks putih.
   - **Kuning**: Kendaraan Angkutan Umum / Barang — di-invert serta didukung ekstraksi kanal kontras kuning.
   - **Merah**: Kendaraan Dinas Pemerintah — polaritas disesuaikan otomatis terhadap kecerahan teks.

2. **Segmentasi Dual-ROI & Validasi Status Pajak (`MM.YY`)**
   - Memisahkan **ROI Nomor Utama** (`roi_cleaned`) dari **ROI Angka Pajak** (`tax_roi_binary` & `tax_roi_gray` di area tengah-bawah pelat) agar teks dudukan pelat tidak ikut terbaca.
   - Menerapkan koreksi karakter pajak (`G/O/D/Q/C -> 0`, `I/L -> 1`, `Z -> 2`, `S -> 5`, `B -> 8`) dan membandingkan `(Tahun, Bulan)` terhadap tanggal sistem saat ini untuk menentukan status **`AKTIF`** atau **`MATI / KEDALUWARSA`**.

---

## Arsitektur Pipeline (11 Tahap Terintegrasi)

| Tahap | Proses | Modul | Keterangan |
| :---: | :--- | :--- | :--- |
| 1 | Deteksi & Crop ROI | [`Script/detector.py`](Script/detector.py) | YOLOv8s (`imgsz` adaptif `640`/`960`, *auto-threshold*, fallback rotasi `±35°`, `iou=0.45`) |
| 2 | Koreksi Kemiringan (*Deskewing*) | [`Script/detector.py`](Script/detector.py) | Proyeksi horizontal (`±45°`) + `cv2.warpAffine` |
| 3 | Warna HSV & Standar Polaritas | [`Script/detector.py`](Script/detector.py) | Klasifikasi Hitam/Putih/Kuning/Merah & standardisasi latar gelap / teks terang |
| 4 | Grayscale, Upscale & Smoothing | [`Script/preprocessor.py`](Script/preprocessor.py) | Konversi Grayscale, upscale min `160px`, dan Bilateral Filter (`d=5, sigma=35`) |
| 5 | Conditional CLAHE | [`Script/preprocessor.py`](Script/preprocessor.py) | Analisis histogram (`mean`, `std`); CLAHE aktif jika `std < 45` atau `mean < 45` |
| 6 | Unsharp Masking | [`Script/preprocessor.py`](Script/preprocessor.py) | Penajaman tepi huruf (`GaussianBlur` + `addWeighted`) & skor ketajaman Laplacian |
| 7 | Segmentasi Biner | [`Script/binarizer.py`](Script/binarizer.py) | Otsu & Adaptive Thresholding (latar `0`, karakter `255`) |
| 8 | Perbaikan Morfologi | [`Script/binarizer.py`](Script/binarizer.py) | *Morphological Closing* pada karakter putih (`255`) untuk menyambung celah |
| 9 | Segmentasi Dual-ROI | [`Script/binarizer.py`](Script/binarizer.py) | Pemisahan `roi_cleaned` (Connected Components) dan `tax_roi_binary` / `tax_roi_gray` |
| 10 | Multi-Input OCR & Validasi Pajak | [`Script/recognizer.py`](Script/recognizer.py) | Voting 3+ kandidat (`Otsu`, `Adaptive`, `Sharp`, `StretchCLAHE`), koreksi posisional, & `recognize_tax()` |
| 11 | Visualisasi Studio Split-Pane | [`Script/utils.py`](Script/utils.py), [`app.py`](app.py) | Antarmuka Streamlit Dark Precision Studio, KPI strip, replika pelat, tabel voting OCR, & grid evaluasi 10 tahap |

---

## Struktur Direktori

```text
ALPR/
├── .streamlit/
│   └── config.toml                # Konfigurasi tema Dark Precision Streamlit
├── Script/
│   ├── __init__.py
│   ├── detector.py                # Tahap 1-3: YOLOv8s auto-threshold, deskewing, HSV & polaritas
│   ├── preprocessor.py            # Tahap 4-6: Grayscale, upscale, Bilateral, CLAHE, Unsharp Masking
│   ├── binarizer.py               # Tahap 7-9: Otsu/Adaptive, Morph Closing, Dual-ROI Utama & Pajak
│   ├── recognizer.py              # Tahap 10 : Multi-input OCR voting, koreksi posisional & status pajak
│   ├── plate_tax_pipeline.py      # Pipeline referensi evaluasi batch nomor plat & pajak
│   └── utils.py                   # Tahap 11 : Anotasi bounding box, badge status & ground truth lookup
├── models/
│   └── best.pt                    # Bobot YOLOv8s terlatih (imgsz=960)
├── app.py                         # Aplikasi Web Streamlit (Split-Pane Inspector & 10-Stage PCD Grid)
├── evaluate_plate_tax.py          # Skrip evaluasi metrik batch pada dataset Kaggle
├── requirements.txt               # Dependensi Python
└── README.md
```

---

## Hasil Evaluasi (Dataset Kaggle `juanthomaswijaya/indonesian-license-plate-dataset`)

- **Deteksi Pelat YOLOv8s (`test` split, 100 gambar / 197 pelat):**
  - Precision: `93.71%`
  - Recall: `98.37%`
  - mAP@0.5: `97.87%`
  - mAP@0.5:0.95: `65.06%`
- **Evaluasi Batch OCR (`evaluate_plate_tax.py`, 30 sampel `test`):**
  - Exact Match Accuracy: `64.00%`
  - Character Accuracy: `91.10%` (`CER: 8.90%`, `F1-Score: 88.49%`)
  - Pajak Valid Terbaca: `36.00%`

---

## Cara Menjalankan

1. **Pasang dependensi:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Jalankan antarmuka Streamlit:**
   ```bash
   streamlit run app.py
   ```
   Buka `http://localhost:8501` di browser.
3. **Evaluasi batch pada dataset Kaggle (opsional):**
   ```bash
   python evaluate_plate_tax.py --split test --max-samples 30 --no-gpu
   ```
4. **Uji modul secara mandiri:**
   ```bash
   python Script/detector.py path/ke/gambar.jpg
   python Script/preprocessor.py path/ke/crop_plat.jpg
   python Script/binarizer.py path/ke/crop_plat.jpg
   python Script/recognizer.py
   ```
