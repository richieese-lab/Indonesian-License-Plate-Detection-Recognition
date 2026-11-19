# Masukan Teknis & Evaluasi Pipeline ALPR (Untuk Tim)

Dokumen ini merangkum hasil evaluasi kuantitatif pada dataset Kaggle (`juanthomaswijaya/indonesian-license-plate-dataset`) setelah pembaruan pada **`Script/preprocessor.py`** dan **`Script/binarizer.py`**, serta rekomendasi teknis spesifik untuk modul **`Script/recognizer.py`**, **`Script/detector.py`**, dan **`app.py`**.

---

## 1. Ringkasan Hasil Evaluasi (Anti-Overfitting: `train` -> `val` -> `test`)

Seluruh pengembangan dan penyesuaian parameter di `preprocessor.py` dan `binarizer.py` dilakukan pada split **`train`** (158 pelat) dan divalidasi pada split **`val`** (179 pelat). Split **`test`** (100 citra / 196 pelat terdeteksi dari 197 ground truth) hanya digunakan satu kali di akhir sebagai pengujian *held-out*.

| Split Dataset | Metrik | Sebelum Perbaikan | Sesudah Perbaikan | Kenaikan |
| :--- | :--- | :---: | :---: | :---: |
| **`train`** (158 pelat) | Exact Match | `32.91%` (52/158) | **`49.37%`** (78/158) | `+16.46%` |
| | Character Accuracy | `70.62%` | **`80.50%`** | `+9.88%` |
| | Pajak Valid (`MM.YY`) | `43.04%` | **`52.53%`** | `+9.49%` |
| **`val`** (179 pelat) | Exact Match | `45.81%` (82/179) | **`53.63%`** (96/179) | `+7.82%` |
| | Character Accuracy | `79.71%` | **`83.50%`** | `+3.79%` |
| | Pajak Valid (`MM.YY`) | `49.72%` | **`59.22%`** | `+9.50%` |
| **`test`** (196 pelat, *held-out*) | **Exact Match** | `37.24%` (73/196) | **`48.47%`** (95/196) | **`+11.23%`** |
| | **Character Accuracy** | `72.88%` | **`81.30%`** | **`+8.42%`** |
| | **Character Error Rate (CER)** | `27.12%` | **`18.70%`** | **`-8.42%`** |
| | **Format Valid Regex** | `83.67%` (164/196) | **`93.37%`** (183/196) | **`+9.70%`** |
| | **Pajak Valid (`MM.YY`)** | `38.27%` (75/196) | **`54.08%`** (106/196) | **`+15.81%`** |

*(Deteksi YOLOv8s `models/best.pt` pada split `test`: Precision `93.71%`, Recall `98.37%`, F1 `95.99%`, mAP@0.5 `97.87%`, mAP@0.5:0.95 `65.06%`.)*

---

## 2. Masukan Teknis untuk Modul `Script/recognizer.py` (Tahap 10 — OCR, Regex, & Voting)

Bagian ini memiliki potensi kenaikan akurasi terbesar berikutnya (berpotensi mendorong *Exact Match* ke `60%+`):

### 2.1. Perbaiki Deteksi Polaritas di `_prepare_for_ocr()` (baris 39–48)
* **Masalah:** Logika `if float(np.mean(image)) < 110.0: return cv2.bitwise_not(image)` sangat rapuh ketika pelat terkena pantulan cahaya (*glare*). Pada citra gelap yang memiliki bercak cahaya, nilai `mean` sering melewati `110` (misal `121.3`), sehingga citra gagal di-invert dan EasyOCR menerima **teks putih di latar abu-abu**, menyebabkan *confidence* anjlok ke ~20–30%.
* **Saran:** Tentukan inversi berdasarkan **median piksel tepi/border** (atau terima parameter polaritas eksplisit), bukan rata-rata seluruh citra.

### 2.2. Tambahkan *Whitelist* Kode Wilayah Pelat Indonesia di `parse_and_validate_plate()`
* **Masalah:** Regex `([A-Z]{1,2})([0-9]{1,4})([A-Z]{1,3})` saat ini meloloskan huruf awal yang **tidak ada di Indonesia** (misalnya `I`, `Q`, `C`, `O`, `J`, `X`, `IB`, `EF`). Akibatnya, sisa garis bingkai kiri yang terbaca `I` di depan `B 1241 FBJ` menjadi `IB 1241 FBJ` dan tetap dianggap valid regex.
* **Saran:** Validasi prefix wilayah terhadap daftar resmi pelat Indonesia:
  ```python
  VALID_REGION_CODES = {
      "A", "B", "D", "E", "F", "G", "H", "K", "L", "M", "N", "P", "R", "S", "T", "W", "Z",
      "AA", "AB", "AD", "AE", "AG", "BA", "BB", "BD", "BE", "BG", "BH", "BK", "BL", "BM",
      "BN", "BP", "DA", "DB", "DC", "DD", "DE", "DG", "DH", "DK", "DL", "DM", "DN", "DP",
      "DR", "DS", "DT", "DW", "EA", "EB", "ED", "KA", "KB", "KH", "KT", "KU", "PA", "PB",
  }
  ```

### 2.3. Perbaiki Ambang `conf >= 0.55` & Mekanisme Voting (baris 151 & 172–183)
* **Masalah:** Kandidat yang formatnya sudah benar tetapi `conf` berada di rentang `0.40–0.54` langsung ditandai `valid=False`. Akibatnya, 1 kandidat yang salah baca namun kebetulan memiliki `conf` tinggi (misal `0.95`) bisa mengalahkan 2–3 kandidat lain yang sebenarnya sepakat pada bacaan yang benar.
* **Saran:**
  1. Izinkan kandidat dengan `conf >= 0.35` masuk voting jika **minimal 2 kandidat** menghasilkan bacaan yang sama (*consensus boost*).
  2. Pertimbangkan **voting tingkat karakter (*character-level voting*)** saat kandidat memiliki panjang sama namun berbeda 1 karakter (misal `H8957FE` vs `H8957FF`).

### 2.4. Koreksi Posisional Berbasis *Minimum Edit Cost* (baris 61–73)
* **Masalah:** Loop `for f_len in (1, 2): for b_len in (2, 3, 1)` langsung mengembalikan kecocokan pertama yang lolos regex tanpa menghitung berapa banyak huruf yang dipaksa berubah. Contoh: `G13AG` (dari GT `G134G`) lebih baik diubah `A -> 4` di posisi angka jika mempertimbangkan jumlah substitusi dan konsistensi dengan kandidat lain.
* **Saran:** Evaluasi semua kandidat pemecahan `(f_len, m_len, b_len)` yang kode wilayahnya ada di `VALID_REGION_CODES`, lalu pilih yang memiliki **jumlah substitusi karakter paling sedikit**.

### 2.5. Cegah Baris Pajak Bocor ke `Grayscale Tajam`
* **Masalah:** Di `app.py` (baris 335), `fallback_image=pre_res["sharpened"]` masih memuat baris pajak di bagian bawah, sehingga EasyOCR sering menggabungkannya ke nomor utama (contoh: `EF18870BA 0L225`).
* **Saran:** Di `app.py`, ganti `fallback_image=bin_res["upscaled_gray"]` (yang sudah di-crop ke ROI utama), **atau** di `recognizer._extract_main_characters()` saring blok teks yang pusat vertikalnya berada di bawah `0.72 * img_h`.

### 2.6. Tingkatkan Akurasi OCR Pajak (`recognize_tax`)
* **Masalah:** Kadang `tax_binary` memutus garis tipis pada angka `4` (terbaca `1`, misal `04.25` menjadi `01.25`) atau menutup lubang angka `8` menjadi `0`.
* **Saran:** Beri bobot lebih tinggi pada bacaan dari `tax_gray` dibanding `tax_binary` (atau wajibkan dukungan minimal satu varian grayscale saat voting), agar tidak salah membaca digit pajak.

---

## 3. Masukan Teknis untuk Modul `Script/detector.py` (Tahap 1–3)

1. **Lengkapi TODO `PlateColorClassifier` (Kebaruan 1):**
   * Saat ini `classify_plate_color()` masih mengembalikan `"TODO"`. Melengkapi klasifikasi warna HSV (`Hitam`, `Putih`, `Kuning`, `Merah`) akan mengembalikan fitur Kebaruan 1 pada UI sekaligus membantu `standardize_polarity()` lebih stabil pada pelat kuning/putih yang terkena bayangan.
2. **Perhalus Pemangkasan Tepi Kiri/Kanan di `trim_plate_margins()`:**
   * Pemangkasan atas/bawah sudah bekerja sangat baik (aktif pada 154/196 pelat uji), namun pada sebagian pelat tepi kiri/kanan masih menyisakan garis bingkai tipis (`kiri 0, kanan 0 px`) yang kadang terbaca sebagai `I`/`1`/`L` di tahap OCR.
