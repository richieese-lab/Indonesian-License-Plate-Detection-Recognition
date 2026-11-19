# Hasil Kerja Branch `zulfan-detector`

Branch eksperimen Zulfan. Bercabang dari `main` di commit `f4e12af` (18 Nov 2026).
Semua angka di bawah berasal dari dataset Kaggle *Indonesian License Plate Dataset* (split `val` + `test`).

> **Status terhadap `main`:** setelah branch ini dibuat, `main` dirombak di commit `21536b6`
> (YOLOv8s 960px, `detector.py` dan `app.py` ditulis ulang). Branch ini **belum bisa di-merge langsung**.
> Bagian yang terbukti membantu perlu dipindahkan ke struktur `main` yang baru.

---

## 1. Perubahan

| File | Perubahan | Materi kuliah |
|---|---|---|
| `Script/detector.py` | Deteksi plat terpotong di tepi gambar | Geometri bounding box |
| `Script/detector.py` | Deskew kedua dengan Canny + Hough sebagai konfirmasi dan cadangan integral proyeksi | P12 (Canny), P17 (Hough, integral proyeksi) |
| `Script/plate_tax_pipeline.py` | Buang sisa garis bingkai di ROI nomor utama | P12 (connected component) |
| `Script/plate_tax_pipeline.py` | Cari baris angka pajak dengan integral proyeksi dari tepi Sobel-x | P8/P12 (Sobel), P17 (integral proyeksi) |
| `Script/plate_tax_pipeline.py`, `app.py` | Opsi `skip_deskew` supaya plat tidak diputar dua kali | — |
| `uji_ocr_banding.py` | Skrip uji OCR versi lama vs baru | — |
| `evaluasi_zulfan.ipynb` | Notebook evaluasi lengkap, tinggal Run All | — |

### 1.1 Plat terpotong di tepi gambar
- **Masalah:** plat yang sebagian keluar dari foto tetap ditandai "Lengkap", karena rasio aspek plat mobil yang terpotong mirip plat motor yang utuh.
- **Cara kerja:** YOLO hanya bisa mengotaki bagian yang terlihat, jadi kotaknya pasti menempel tepi gambar. Kalau `x1`/`y1` ≤ 3 px dari 0, atau `x2`/`y2` ≤ 3 px dari lebar/tinggi gambar, status menjadi **"Terpotong di Tepi Gambar"**.
- **Kode:** `PlateDetector.touches_image_edge()`, dipanggil di `detect()`.

### 1.2 Deskew kedua (Canny + Hough)
- **Cara kerja:** Canny mencari tepi, Hough memvoting garis lurus terpanjang (bingkai plat). Sudut garis = sudut plat.
- **Kenapa hanya konfirmasi/cadangan:** di foto asli Hough sering tertipu garis bemper, plat motor yang melengkung, dan perspektif. Integral proyeksi tetap jadi estimasi utama. Hough dipakai kalau proyeksi ragu tetapi garis bingkai sangat jelas (total panjang garis ≥ 5× lebar plat).
- **Kode:** `estimate_skew_hough()`, digabung di `PlateDetector.deskew_plate()`. Status deskew sekarang menyebut sumbernya, misalnya *"Dikoreksi -1.2 derajat (proyeksi & Hough sepakat)"*.

### 1.3 Sisa garis bingkai di ROI nomor utama
- **Masalah:** garis tegak bingkai plat ikut masuk citra biner dan dibaca OCR sebagai "I", "1", atau "L". Contoh: **B 1388 DFX** terbaca **BI 3880 FXL**, karena hasil mentahnya `B1388DFXL` (9 karakter) dipaksa `fix_plate_by_position()` menjadi `B1 | 388D | FXL`.
- **Cara kerja:** komponen yang menempel tepi kiri/kanan ROI **dan** kurus (lebar < 30% tinggi) atau berbentuk sudut "L" (menempel atas/bawah, lebar < 60% tinggi) dibuang sebelum OCR.
- **Kode:** `_is_frame_piece()`, dipakai `_clean_cc()`.

### 1.4 Posisi baris angka pajak
- **Masalah:** ROI pajak diambil dari posisi tetap 66–92% tinggi crop. Kalau kotak YOLO longgar, baris pajak bergeser sehingga angkanya terpotong atau tulisan dealer ikut masuk.
- **Cara kerja:** hitung proporsi tepi tegak (Sobel-x) per baris → integral proyeksi horizontal. Garis bingkai yang mendatar hampir tidak punya gradien-x, jadi profil hanya tinggi di baris tulisan. Pita paling tebal di atas = nomor utama; baris pajak dicari tepat di bawahnya dengan ambang lokal dan **harus ada jeda** di antaranya. Kalau tidak ketemu, kembali ke posisi tetap lama.
- **Kode:** `_edge_row_profile()`, `_text_bands()`, `_find_tax_rows()`, dipakai `process_tax_roi()`.

### 1.5 Deskew ganda
- **Masalah:** `preprocess_plate()` memutar ulang plat dengan Hough padahal sudah diluruskan `detector.py`. Pada 18% plat terjadi putaran kedua, dan sebagian membatalkan koreksi yang benar (contoh: G 1180 AZ, AC 5928 REY).
- **Kode:** parameter `skip_deskew` di `preprocess_plate()` dan `analyze_single_plate()`; `app.py` memakai `skip_deskew=True`.

### 1.6 ROI pajak: batas kiri-kanan + OCR beberapa ROI
- **Masalah (ditemukan dari notebook evaluasi):** dengan YOLO sungguhan, pajak terbaca justru turun dari 32,5% ke 22,5%. Penyebabnya dua: (1) lebar ROI pajak masih tetap 26–74%, padahal di plat motor dan plat model lama angka pajak ada di kanan bawah (contoh 12·16, 09·18), jadi digitnya terpotong; (2) kalau proyeksi baris salah pilih, tidak ada cadangan karena ROI lama tidak dicoba lagi.
- **Cara kerja:**
  - Di dalam pita baris pajak, hitung **integral proyeksi vertikal** dari tepi Sobel-x (P17). Kolom yang aktif dikelompokkan, celah sempit (titik pemisah MM·YY) disambung, lalu dipilih kelompok dengan massa tepi terbesar sebagai batas kiri-kanan.
  - OCR dijalankan pada **tiga ROI**: `Tengah` (posisi tetap lama, selalu ikut), `Proyeksi` (baris + kolom hasil proyeksi), dan `Kolom` (baris bawah tetap + kolom hasil proyeksi). ROI yang hampir sama (IoU ≥ 0,85) cukup dipakai sekali, dan ROI yang blur dilewati.
  - Hasil dipilih dengan **voting**: setiap varian yang menghasilkan format MM-YY yang sama menambah skor `1 + conf`. Angka yang muncul di beberapa ROI lebih dipercaya daripada satu bacaan yang kebetulan conf-nya tinggi.
- **Konsekuensi:** OCR pajak lebih lambat (sekitar 2–3× panggilan EasyOCR untuk baris pajak).
- **Kode:** `_tax_column_range()`, `_tax_variants()`, `tax_roi_candidates()`, `process_tax_roi()`, voting di `recognize_tax_date()`.

### 1.7 Keputusan angka pajak: utamakan tidak salah baca
- **Kasus pemicu (foto parkiran):** B 6722 PWI terbaca 09-10 padahal 09-18; B 6231 NNZ pajaknya tidak terbaca dan crop yang ditampilkan berisi stiker + "16" saja.
- **Penyebab yang ditemukan:**
  - Varian **biner (Otsu)** menutup lubang kecil angka 8 sehingga terbaca 0, dan salahnya sama persis di semua ROI. Voting jadi memilih yang salah karena "sepakat dalam kesalahan".
  - Jendela pencarian baris pajak di `_find_tax_rows()` terlalu pendek (0,8× tinggi huruf utama), jadi bagian bawah angka pajak terpotong.
  - Stiker di kiri baris pajak punya tepi tegak yang kuat sehingga proyeksi kolom memilih stiker, bukan angka pajak.
  - Titik pemisah MM·YY sering terbaca angka ("12816"), dan EasyOCR kadang mengembalikan kotak kosong dengan conf 0 yang menurunkan rata-rata conf.
- **Perbaikan:**
  - Jendela pencarian baris pajak diperpanjang jadi 1,5× tinggi huruf utama.
  - ROI keempat `Kanan` (separuh kanan plat) untuk plat yang angka pajaknya rata kanan.
  - Varian abu-abu tambahan `Gray80` (diperbesar lebih kecil, tanpa binerisasi).
  - Voting: suara varian biner berbobot setengah, dan pemenang **wajib didukung minimal satu varian abu-abu**. Kalau tidak ada, hasilnya "tidak terbaca". Bacaan abu-abu yang ragu (conf 0,45–0,55) tetap diterima kalau minimal dua varian sepakat.
  - Lima digit dengan digit tengah dianggap titik (12816 → 12-16); kotak OCR kosong dibuang sebelum menghitung conf (juga untuk nomor utama).
  - Status pajak menyebut ROI asal bacaan, dan `app.py` menampilkan crop ROI tersebut (`process_tax_roi(..., prefer=...)`).
- **Hasil uji** (EasyOCR asli, 120 plat test yang angka pajaknya bisa dibaca manual, crop longgar mirip YOLO):

| Ukuran plat | Versi | Benar | Salah baca | Tidak terbaca |
|---|---|---|---|---|
| Asli (lebar ±250 px) | sebelum 1.6–1.7 | 44 | 24 | 52 |
| Asli (lebar ±250 px) | sekarang | **47** | **12** | 61 |
| Kecil (lebar 125 px, seperti foto parkiran) | sebelum 1.6–1.7 | 14 | 21 | 85 |
| Kecil (lebar 125 px, seperti foto parkiran) | sekarang | 11 | **12** | 97 |

  Salah baca turun kira-kira setengahnya. Di plat kecil, jumlah yang benar sedikit turun: angka pajak di sana hanya setinggi ±6–8 piksel, di bawah batas yang bisa dibaca EasyOCR dengan andal, jadi sistem lebih sering memilih "tidak terbaca" daripada menebak.
- **Kode:** `_find_tax_rows()`, `tax_roi_candidates()`, `_tax_variants()`, `recognize_tax_date(..., return_source=True)`, `process_tax_roi(..., prefer=None)`, `analyze_single_plate()` (kunci baru `tax_roi`).

---

## 2. Yang diuji tetapi tidak dipakai

| Ide | Hasil uji | Keputusan |
|---|---|---|
| Hough sebagai deskew utama | Sering salah di foto asli (bemper, plat melengkung, perspektif) | Hanya konfirmasi/cadangan |
| Morfologi closing pada mask tulisan sebelum integral proyeksi (P13–P14) | Tidak menaikkan kesepakatan dengan Hough (55/60 tetap), plat lurus malah ikut diputar | Tidak dipakai |
| Potong ulang plat berbasis kontur (P12, P13–P14) | Memotong huruf (S 1570 JH) dan baris pajak (L 9714 BJ); separuh kasus kontur tidak ketemu | Tidak dipakai, tetap pakai rumus geometri |

---

## 3. Hasil uji

### 3.1 Tahap detector
| Uji | Hasil |
|---|---|
| 48 plat sintetis (4 warna × 4 sudut × 3 kondisi cahaya): sudut deskew ±2° | 48/48 |
| 384 plat sintetis (mobil/motor, italic, noise): galat sudut rata-rata | 0.04° |
| Plat asli yang dikoreksi: integral proyeksi & Hough sepakat (selisih ≤ 2.5°) | **155/160 (97%)** |
| Plat asli yang sebelumnya dibiarkan miring dan sekarang diluruskan oleh cadangan Hough | 3 plat |
| Perubahan hasil klasifikasi warna akibat perubahan deskew | 0 plat |

### 3.2 Tahap OCR — pengukuran cacat citra (tanpa OCR, 376 plat, alur aplikasi)
| Ukuran | Versi lama | Versi baru |
|---|---|---|
| Sisa garis bingkai di ROI utama | 34% | **0%** |
| Jumlah komponen = jumlah karakter label | 28% | **45%** |
| Digit pajak menempel tepi ROI (terpotong) | 44% | **26%** |
| Plat diputar dua kali | 18% | **0%** |

Pada uji visual baris pajak (crop longgar mirip kotak YOLO, 60 plat), sekitar 15 plat membaik dan 0 memburuk setelah syarat "harus ada jeda" ditambahkan.
Efek samping yang diketahui: 3 plat memburuk karena huruf terakhirnya menempel ke bingkai sehingga ikut terhapus (W 1518 WF, B 9374 RI, L 4561 VK), dibanding ±109 plat yang membaik.

### 3.3 Tahap OCR — EasyOCR sungguhan (`uji_ocr_banding.py`, 40 plat test, kotak longgar mirip YOLO)
| Metrik | Versi lama | Versi baru |
|---|---|---|
| Nomor plat tepat (exact match) | 11/40 (28%) | **14/40 (35%)** |
| Character Error Rate (CER) | 62.4% | **56.4%** |
| Angka pajak terbaca format MM-YY | 11/40 | **12/40** (sebelum perbaikan 1.6) |

Catatan: angka ini dihitung dari **semua plat** (plat yang ditolak "TIDAK VALID" dihitung salah), jadi tidak bisa dibandingkan langsung dengan angka README `main` yang hanya menghitung plat yang lolos validasi.

---

## 4. Cara menjalankan

1. **Notebook evaluasi lengkap:** buka `evaluasi_zulfan.ipynb`, pilih kernel `venv`, lalu **Run All**.
   Isinya: validasi YOLO (confusion matrix, kurva F1, PR, mAP), statistik tahap detector, dan OCR lama vs baru
   (exact match, CER, precision/recall/F1 karakter, confusion matrix karakter). Hasil tersimpan di `Output/evaluasi/<tanggal-jam>/`.
2. **Uji OCR cepat dari terminal:** `python uji_ocr_banding.py` (opsi `--n 100` untuk lebih banyak plat).
3. **Aplikasi:** `streamlit run app.py`.
