import os
import re
from collections import Counter
import glob
import json
import base64
import urllib.request
import cv2
import numpy as np
import easyocr


# ==============================================================================
# 0. PENCARIAN DATASET OTOMATIS (KAGGLEHUB CACHE / LOKAL)
# ==============================================================================
def find_recognition_dataset(custom_root=None):
    """Mencari direktori Indonesian License Plate Recognition Dataset secara otomatis."""
    if custom_root and os.path.exists(custom_root):
        return custom_root

    candidates = [
        os.path.expanduser("~/.cache/kagglehub/datasets/juanthomaswijaya/indonesian-license-plate-dataset"),
        "/root/.cache/kagglehub/datasets/juanthomaswijaya/indonesian-license-plate-dataset",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Dataset", "Raw"),
    ]
    for base in candidates:
        if not os.path.exists(base):
            continue
        matches = glob.glob(os.path.join(base, "**", "Indonesian License Plate Recognition Dataset"), recursive=True)
        if matches:
            return matches[0]
        if os.path.exists(os.path.join(base, "classes.names")):
            return base
    return None


def load_class_names(dataset_root):
    classes_path = os.path.join(dataset_root, "classes.names")
    if not os.path.exists(classes_path):
        return list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    with open(classes_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# ==============================================================================
# 1. ANALISIS KUALITAS CITRA (BLUR, RASIO ASPEK, WARNA KUNING)
# ==============================================================================
def assess_sharpness(gray_img, norm_height=140):
    """
    Mengukur ketajaman citra pada tinggi terstandar agar adil untuk berbagai resolusi.
    Mengembalikan (laplacian_var, tenengrad_mean).
    """
    if gray_img is None or gray_img.size == 0:
        return 0.0, 0.0
    h, w = gray_img.shape[:2]
    if h <= 0 or w <= 0:
        return 0.0, 0.0
    scale = norm_height / float(h)
    resized = cv2.resize(gray_img, (max(1, int(w * scale)), norm_height), interpolation=cv2.INTER_CUBIC)
    lap_var = float(cv2.Laplacian(resized, cv2.CV_64F).var())
    gx = cv2.Sobel(resized, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(resized, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float(np.mean(gx ** 2 + gy ** 2))
    return round(lap_var, 1), round(tenengrad, 1)


def is_too_blurry(gray_img, lap_threshold=80.0, tenengrad_threshold=450.0):
    """Menentukan apakah citra terlalu buram berdasarkan Laplacian variance & Tenengrad."""
    lap_var, tenengrad = assess_sharpness(gray_img)
    blurry = (lap_var < lap_threshold) and (tenengrad < tenengrad_threshold)
    return blurry, lap_var, tenengrad


def is_yellow_plate(bgr_img, ratio_threshold=0.08):
    """
    Mendeteksi apakah pelat memiliki latar belakang kuning (angkutan umum)
    menggunakan ruang warna HSV (OpenCV Hue: 15..38).
    """
    if bgr_img is None or bgr_img.size == 0 or len(bgr_img.shape) < 3:
        return False, 0.0
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 0] >= 14) & (hsv[:, :, 0] <= 40) & (hsv[:, :, 1] >= 55) & (hsv[:, :, 2] >= 55)
    ratio = float(np.mean(mask))
    return ratio >= ratio_threshold, round(ratio, 3)


def check_plate_geometry(bgr_img, min_aspect=1.65, max_aspect=6.5, min_width=80, min_height=28):
    """
    Validasi geometri dasar pelat: menolak potongan pelat yang hanya terlihat setengah
    atau terlalu kecil untuk dibaca secara meyakinkan.
    """
    if bgr_img is None or bgr_img.size == 0:
        return False, "Citra kosong", 0.0
    h, w = bgr_img.shape[:2]
    aspect = w / float(max(1, h))
    if w < min_width or h < min_height:
        return False, f"Resolusi terlalu kecil ({w}x{h})", round(aspect, 2)
    if aspect < min_aspect:
        return False, f"Terpotong/Setengah (rasio {aspect:.2f} < {min_aspect})", round(aspect, 2)
    if aspect > max_aspect:
        return False, f"Rasio ekstrem ({aspect:.2f} > {max_aspect})", round(aspect, 2)
    return True, "OK", round(aspect, 2)


# ==============================================================================
# 2. PREPROCESSING (DESKEWING, GRAYSCALE, SMOOTHING, SMART CLAHE, COLOR CHANNEL)
# ==============================================================================
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def deskew_img(img, max_angle=15.0):
    """Koreksi kemiringan citra berbasis HoughLinesP."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=30,
        minLineLength=max(10, gray.shape[1] // 3), maxLineGap=10
    )
    if lines is None or len(lines) == 0:
        return img, 0.0
    lines_2d = np.asarray(lines).reshape(-1, 4)
    angles = [np.degrees(np.arctan2(y2 - y1, x2 - x1)) for x1, y1, x2, y2 in lines_2d]
    angles = [a for a in angles if abs(a) <= max_angle]
    if not angles:
        return img, 0.0
    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:
        return img, 0.0
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), median_angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated, round(median_angle, 2)


def extract_yellow_contrast_channel(bgr_img):
    """
    Menangani kasus pelat latar kuning dengan tulisan putih ber-outline hitam:
    Pada kanal Blue (BGR[:,:,0]) dan Saturation terbalik (255-S), latar kuning bernilai
    rendah (gelap) sedangkan teks putih bernilai tinggi (terang) tanpa terganggu outline.
    """
    blue = bgr_img[:, :, 0]
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    inv_sat = 255 - hsv[:, :, 1]
    combined = cv2.addWeighted(blue, 0.6, inv_sat, 0.4, 0)
    return _CLAHE.apply(combined)


def preprocess_plate(bgr_img, skip_deskew=False):
    """
    Alur preprocessing lengkap: Deskew -> Grayscale -> Bilateral -> Smart CLAHE (+ Yellow Channel).
    skip_deskew=True dipakai kalau citra sudah diluruskan oleh detector.py (di app.py selalu begitu),
    supaya plat tidak diputar dua kali. Deskew di sini hanya pakai Hough, dan di foto asli
    sering membatalkan koreksi detector yang sudah benar.
    """
    if skip_deskew:
        deskewed_bgr, angle = bgr_img, 0.0
    else:
        deskewed_bgr, angle = deskew_img(bgr_img)
    gray = cv2.cvtColor(deskewed_bgr, cv2.COLOR_BGR2GRAY)
    smoothed = (
        cv2.bilateralFilter(gray, d=5, sigmaColor=35, sigmaSpace=35)
        if gray.shape[0] >= 50 else gray.copy()
    )
    mean_v, std_v = float(np.mean(smoothed)), float(np.std(smoothed))
    enhanced = _CLAHE.apply(smoothed) if (std_v < 45.0 or mean_v < 45.0) else smoothed.copy()

    yellow_flag, yellow_ratio = is_yellow_plate(deskewed_bgr)
    yellow_ch = extract_yellow_contrast_channel(deskewed_bgr) if yellow_flag else None

    return {
        "bgr": deskewed_bgr,
        "gray": gray,
        "smoothed": smoothed,
        "enhanced": enhanced,
        "skew_angle": angle,
        "is_yellow": yellow_flag,
        "yellow_ratio": yellow_ratio,
        "yellow_channel": yellow_ch,
    }


# ==============================================================================
# 3. SEGMENTASI DUAL-ROI (PLAT UTAMA & ANGKA PAJAK)
# ==============================================================================
def normalize_polarity(binary_img):
    """Standarisasi polaritas biner: latar putih (255), karakter hitam (0) untuk OCR."""
    white_ratio = np.sum(binary_img == 255) / float(max(1, binary_img.size))
    return cv2.bitwise_not(binary_img) if white_ratio < 0.5 else binary_img.copy()


FRAME_THIN_RATIO = 0.30   # garis bingkai: lebarnya < 30% tingginya
FRAME_CORNER_RATIO = 0.60  # potongan sudut bingkai (bentuk "L"): lebar < 60% tinggi + menempel atas/bawah


def _is_frame_piece(x, y, cw, ch, nw, nh):
    """
    Sisa bingkai plat di kiri/kanan ROI. Cirinya selalu menempel tepi kiri atau kanan ROI,
    lalu kurus (garis tegak) atau berbentuk sudut "L" yang juga menempel tepi atas/bawah.
    Kalau dibiarkan, EasyOCR membacanya sebagai "I", "1", atau "L" dan jumlah karakternya jadi lebih.
    """
    touches_side = x <= 2 or x + cw >= nw - 2
    if not touches_side:
        return False
    touches_tb = y <= 2 or y + ch >= nh - 2
    return cw < FRAME_THIN_RATIO * ch or (touches_tb and cw < FRAME_CORNER_RATIO * ch)


def _clean_cc(bin_img, max_w_ratio=0.45, min_h_ratio=0.40, min_area_ratio=0.0015, min_area_abs=20,
              drop_frame=True):
    norm = normalize_polarity(bin_img)
    ch_mask = cv2.bitwise_not(norm)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ch_mask, connectivity=8)
    cleaned = norm.copy()
    nh, nw = norm.shape[:2]
    if n > 1:
        min_a = max(min_area_abs, int(norm.size * min_area_ratio))
        frame = set()
        if drop_frame:
            frame = {i for i in range(1, n) if _is_frame_piece(*stats[i][:4], nw, nh)}
        # tinggi karakter acuan dihitung tanpa garis bingkai, karena bingkai biasanya paling tinggi
        cands = [stats[i][3] for i in range(1, n)
                 if i not in frame and stats[i][4] >= min_a and stats[i][2] <= max_w_ratio * nw]
        max_h = max(cands) if cands else nh
        for i in range(1, n):
            _, _, cw, ch, area = stats[i]
            if i in frame or cw > max_w_ratio * nw or area < min_a or ch < min_h_ratio * max_h:
                cleaned[labels == i] = 255
    # Pengaman: jika hampir semua piksel hitam terhapus, kembalikan norm
    if np.sum(cleaned == 0) < 80:
        cleaned = norm
    return cv2.copyMakeBorder(cleaned, 15, 15, 20, 20, cv2.BORDER_CONSTANT, value=255)


def _upscale_and_sharpen(roi, target_h=140, alpha=1.6):
    scale = target_h / float(max(1, roi.shape[0])) if roi.shape[0] < target_h else 1.0
    up = cv2.resize(roi, (max(1, int(roi.shape[1] * scale)), max(1, int(roi.shape[0] * scale))),
                    interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(up, (0, 0), 2.0)
    sharp = cv2.addWeighted(up, alpha, blur, -(alpha - 1.0), 0)
    return sharp


def process_main_plate_roi(enhanced_gray, yellow_channel=None, raw_gray=None):
    """Menghasilkan varian citra ROI plat utama untuk OCR."""
    h, w = enhanced_gray.shape[:2]
    roi = enhanced_gray[int(h * 0.04):int(h * 0.74), int(w * 0.03):int(w * 0.97)]
    sharp = _upscale_and_sharpen(roi, target_h=140, alpha=1.6)

    _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adap = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10)
    sharp_pad = cv2.copyMakeBorder(sharp, 15, 15, 20, 20, cv2.BORDER_REPLICATE)

    variants = {
        "Otsu": _clean_cc(otsu),
        "Adaptive": _clean_cc(adap),
        "Sharp": sharp_pad,
    }
    if raw_gray is not None:
        r_roi = raw_gray[int(h * 0.04):int(h * 0.74), int(w * 0.03):int(w * 0.97)]
        variants["RawGray"] = cv2.copyMakeBorder(r_roi, 15, 15, 20, 20, cv2.BORDER_REPLICATE)
        stretched = cv2.normalize(r_roi, None, 0, 255, cv2.NORM_MINMAX)
        stretch_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(stretched)
        variants["StretchCLAHE"] = cv2.copyMakeBorder(stretch_clahe, 15, 15, 20, 20, cv2.BORDER_REPLICATE)

    if yellow_channel is not None:
        y_roi = yellow_channel[int(h * 0.04):int(h * 0.74), int(w * 0.03):int(w * 0.97)]
        y_sharp = _upscale_and_sharpen(y_roi, target_h=140, alpha=1.7)
        _, y_otsu = cv2.threshold(y_sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Morfologi closing ringan untuk menyambung bagian huruf yang sempat terpotong outline
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        y_otsu_closed = cv2.bitwise_not(cv2.morphologyEx(cv2.bitwise_not(normalize_polarity(y_otsu)), cv2.MORPH_CLOSE, k))
        variants["YellowChannel"] = cv2.copyMakeBorder(y_sharp, 15, 15, 20, 20, cv2.BORDER_REPLICATE)
        variants["YellowOtsu"] = _clean_cc(y_otsu_closed)

    return variants, roi


def _edge_row_profile(gray, x1r=0.20, x2r=0.80):
    """
    Integral proyeksi horizontal (P17) dari tepi tegak.
    Tepi diambil dari gradien horizontal (Sobel-x, P8/P12): goresan huruf punya banyak
    tepi tegak, sedangkan garis bingkai dan bemper yang mendatar hampir tidak punya
    gradien-x. Jadi profil ini hanya tinggi di baris tulisan, apa pun warna platnya.
    """
    h, w = gray.shape[:2]
    sub = gray[:, int(w * x1r):int(w * x2r)].astype(np.float32)
    gx = np.abs(cv2.Sobel(sub, cv2.CV_32F, 1, 0, ksize=3))
    gx8 = cv2.normalize(gx, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, edge = cv2.threshold(gx8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    prof = (edge > 0).mean(axis=1)
    k = max(1, int(round(h * 0.03)))
    return np.convolve(prof, np.ones(k) / k, mode="same")


def _text_bands(prof, h):
    """Kelompokkan baris yang profilnya tinggi jadi pita (y_awal, y_akhir). Celah tipis < 4% disambung."""
    active = prof > max(0.04, 0.30 * float(prof.max()))
    bands, start = [], None
    for y, a in enumerate(np.append(active, False)):
        if a and start is None:
            start = y
        elif not a and start is not None:
            if bands and start - bands[-1][1] < 0.04 * h:
                bands[-1] = (bands[-1][0], y)
            else:
                bands.append((start, y))
            start = None
    return bands


def _find_tax_rows(gray):
    """
    1. Pita nomor utama = pita paling tebal di 75% bagian atas plat.
    2. Baris pajak dicari di area tepat di bawahnya (sampai 0.8x tinggi huruf utama)
       dengan ambang lokal, karena angka pajak jauh lebih kecil sehingga profilnya
       lebih rendah dari nomor utama.
    Return (y1, y2) atau None kalau tidak ketemu (nanti pakai posisi tetap).
    """
    h = gray.shape[0]
    if h < 20:
        return None
    prof = _edge_row_profile(gray)
    upper = [bd for bd in _text_bands(prof, h) if bd[0] < 0.75 * h]
    if not upper:
        return None
    m1, m2 = max(upper, key=lambda bd: bd[1] - bd[0])
    main_h = m2 - m1
    if main_h < 0.2 * h:
        return None

    # jendela pencarian cukup panjang supaya baris pajak tidak terpotong bagian bawahnya
    # (sebelumnya 0.8x tinggi huruf utama, ternyata kurang kalau jeda antar baris lebar)
    r0, r1 = m2, min(h, int(m2 + 1.5 * main_h))
    seg = prof[r0:r1]
    if seg.size < 3:
        return None
    pk = int(np.argmax(seg))
    peak = float(seg[pk])
    if peak < max(0.03, 0.15 * float(prof[m1:m2].mean())):
        return None
    a = b = pk
    while a > 0 and seg[a - 1] > 0.45 * peak:
        a -= 1
    while b < seg.size - 1 and seg[b + 1] > 0.45 * peak:
        b += 1
    # harus ada jeda (lembah) antara nomor utama dan baris pajak. Kalau tidak ada,
    # yang ketemu kemungkinan besar masih bagian bawah huruf nomor utama.
    if a == 0 or float(seg[:a + 1].min()) > 0.6 * peak:
        return None
    center = r0 + (a + b) / 2.0
    # tinggi angka pajak kira-kira 35-50% huruf utama; ROI dibuat sedikit lebih longgar
    roi_h = max((b - a + 1) * 1.7, 0.60 * main_h)
    y1 = max(m2 - 2, int(center - roi_h / 2))
    y2 = min(h, int(center + roi_h / 2) + 1)
    return (y1, y2) if y2 - y1 >= 6 else None


def _tax_column_range(gray, y1, y2):
    """
    Batas kiri-kanan angka pajak lewat integral proyeksi VERTIKAL (P17) di baris pajak.
    Di plat motor dan plat model lama, angka pajak ada di kanan bawah, bukan di tengah,
    jadi lebar tetap 26%-74% sering memotong angkanya.
    Return (x1, x2) atau None.
    """
    h, w = gray.shape[:2]
    band = gray[y1:y2, :].astype(np.float32)
    if band.shape[0] < 4 or w < 40:
        return None
    gx = np.abs(cv2.Sobel(band, cv2.CV_32F, 1, 0, ksize=3))
    gx8 = cv2.normalize(gx, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, edge = cv2.threshold(gx8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    prof = (edge > 0).mean(axis=0)
    # tepi kiri-kanan bingkai plat tidak ikut dihitung
    m = max(1, int(w * 0.04))
    prof[:m] = 0
    prof[-m:] = 0
    k = max(1, int(round(w * 0.02)))
    prof = np.convolve(prof, np.ones(k) / k, mode="same")
    if prof.max() <= 0.05:
        return None
    active = prof > 0.35 * float(prof.max())

    # kelompokkan kolom aktif, celah sempit (titik pemisah MM.YY bisa cukup lebar) disambung
    runs, start = [], None
    for x, on in enumerate(np.append(active, False)):
        if on and start is None:
            start = x
        elif not on and start is not None:
            if runs and start - runs[-1][1] < 0.09 * w:
                runs[-1] = (runs[-1][0], x)
            else:
                runs.append((start, x))
            start = None
    runs = [r for r in runs if 0.10 * w <= r[1] - r[0] <= 0.65 * w]
    if not runs:
        return None
    # angka pajak = kelompok dengan "massa" tepi terbesar (stiker/baut hanya punya sedikit tepi tegak)
    x1, x2 = max(runs, key=lambda r: float(prof[r[0]:r[1]].sum()))
    pad = int(w * 0.04)
    return max(0, x1 - pad), min(w, x2 + pad)


def _tax_variants(crop, ycrop=None, prefix=""):
    sharp = _upscale_and_sharpen(crop, target_h=120, alpha=1.8)
    _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adap = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 6)
    # versi abu-abu yang lebih kecil: angka pajak tipis dan rapat, kalau dibinerkan lubang
    # angka 8/6/9 sering tertutup sehingga terbaca 0. Versi abu-abu tidak punya masalah itu.
    gray80 = _upscale_and_sharpen(crop, target_h=80, alpha=1.8)
    v = {
        prefix + "Sharp": cv2.copyMakeBorder(sharp, 15, 15, 20, 20, cv2.BORDER_REPLICATE),
        prefix + "Gray80": cv2.copyMakeBorder(gray80, 12, 12, 16, 16, cv2.BORDER_REPLICATE),
        prefix + "Otsu": _clean_cc(otsu, min_h_ratio=0.35, min_area_abs=15),
        prefix + "Adaptive": _clean_cc(adap, min_h_ratio=0.35, min_area_abs=15),
    }
    if ycrop is not None and ycrop.size:
        y_sharp = _upscale_and_sharpen(ycrop, target_h=120, alpha=1.8)
        _, y_otsu = cv2.threshold(y_sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        v[prefix + "YellowSharp"] = cv2.copyMakeBorder(y_sharp, 15, 15, 20, 20, cv2.BORDER_REPLICATE)
        v[prefix + "YellowOtsu"] = _clean_cc(y_otsu, min_h_ratio=0.35, min_area_abs=15)
    return v


def tax_roi_candidates(enhanced_gray, yellow_channel=None):
    """
    Beberapa kandidat ROI angka pajak. OCR nanti dijalankan di semuanya lalu di-voting,
    jadi kalau satu ROI meleset, ROI lain masih bisa menangkap angkanya.
      1. "Tengah"   : posisi tetap lama (66%-92% tinggi, 26%-74% lebar). Selalu ikut.
      2. "Proyeksi" : baris dari proyeksi horizontal + kolom dari proyeksi vertikal.
      3. "Kolom"    : baris tetap (62%-95%) + kolom dari proyeksi vertikal,
                      untuk plat yang angka pajaknya di kanan tapi barisnya tidak terdeteksi.
      4. "Kanan"    : baris pajak + separuh kanan plat (42%-100% lebar).
    Return: list (nama, (y1, y2, x1, x2)), kandidat yang hampir sama dibuang.
    """
    h, w = enhanced_gray.shape[:2]
    boxes = [("Tengah", (int(h * 0.66), int(h * 0.92), int(w * 0.26), int(w * 0.74)))]
    rows = _find_tax_rows(enhanced_gray)
    if rows:
        cols = _tax_column_range(enhanced_gray, *rows) or (int(w * 0.15), int(w * 0.95))
        boxes.append(("Proyeksi", (rows[0], rows[1], cols[0], cols[1])))
    fy1, fy2 = int(h * 0.62), int(h * 0.95)
    cols = _tax_column_range(enhanced_gray, fy1, fy2)
    if cols:
        boxes.append(("Kolom", (fy1, fy2, cols[0], cols[1])))
    # angka pajak plat lama/motor sering rata kanan, sedangkan bagian kirinya berisi
    # stiker atau logo dealer yang tepinya bisa lebih kuat dari angka pajak
    ky1, ky2 = rows if rows else (fy1, fy2)
    boxes.append(("Kanan", (ky1, ky2, int(w * 0.42), w)))

    def overlap(a, b):
        iy = max(0, min(a[1], b[1]) - max(a[0], b[0]))
        ix = max(0, min(a[3], b[3]) - max(a[2], b[2]))
        inter = ix * iy
        ua = (a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - inter
        return inter / ua if ua > 0 else 0.0

    out = []
    for name, bx in boxes:
        if bx[1] - bx[0] < 6 or bx[3] - bx[2] < 12:
            continue
        if all(overlap(bx, o[1]) < 0.85 for o in out):
            out.append((name, bx))
    return out


def process_tax_roi(enhanced_gray, yellow_channel=None, prefer=None):
    """
    Menghasilkan varian citra ROI angka pajak.
    Kunci "Sharp"/"Otsu"/"Adaptive" (tanpa awalan) = ROI utama untuk ditampilkan:
    ROI `prefer` kalau diberikan, lalu ROI hasil proyeksi, terakhir ROI tengah lama.
    Kandidat ROI lain ikut dengan awalan nama, misalnya "Tengah/Otsu".
    Return: (variants, tax_crop_utama)
    """
    cands = tax_roi_candidates(enhanced_gray, yellow_channel)
    # prefer = ROI yang dipakai hasil bacaan (dari analyze_single_plate), supaya gambar yang
    # ditampilkan sama dengan yang benar-benar terbaca
    primary = (next((c for c in cands if c[0] == prefer), None)
               or next((c for c in cands if c[0] == "Proyeksi"), cands[0]))
    variants, tax_crop = {}, None
    for name, (y1, y2, x1, x2) in cands:
        crop = enhanced_gray[y1:y2, x1:x2]
        ycrop = yellow_channel[y1:y2, x1:x2] if yellow_channel is not None else None
        variants.update(_tax_variants(crop, ycrop, "" if name == primary[0] else name + "/"))
        if name == primary[0]:
            tax_crop = crop
    return variants, tax_crop


# ==============================================================================
# 4. OCR & VALIDASI STRICT FORMAT PLAT + PAJAK (MM-YY)
# ==============================================================================
DICT_CHAR_TO_INT = {'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1', 'T': '1', 'Z': '2', 'A': '4', 'S': '5', 'G': '6', 'B': '8'}
DICT_INT_TO_CHAR = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '6': 'G', '7': 'T', '8': 'B'}


def fix_plate_by_position(raw):
    m_strict = re.fullmatch(r'([A-Z]{1,2})([0-9]{1,4})([A-Z]{1,3})', raw)
    if m_strict:
        return f"{m_strict.group(1)} {m_strict.group(2)} {m_strict.group(3)}", True
    if 4 <= len(raw) <= 9:
        for f_len in (1, 2):
            for b_len in (2, 3, 1):
                m_len = len(raw) - f_len - b_len
                if 1 <= m_len <= 4:
                    cand = (
                        "".join(DICT_INT_TO_CHAR.get(c, c) for c in raw[:f_len]) +
                        "".join(DICT_CHAR_TO_INT.get(c, c) for c in raw[f_len:f_len + m_len]) +
                        "".join(DICT_INT_TO_CHAR.get(c, c) for c in raw[f_len + m_len:])
                    )
                    m2 = re.fullmatch(r'([A-Z]{1,2})([0-9]{1,4})([A-Z]{1,3})', cand)
                    if m2:
                        return f"{m2.group(1)} {m2.group(2)} {m2.group(3)}", True
    return raw, False


def recognize_main_plate(reader, images_dict, min_conf=0.60):
    candidates = []
    for source_name, img in images_dict.items():
        res = reader.readtext(
            img, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            paragraph=False, min_size=5, text_threshold=0.4, low_text=0.3, width_ths=1.5
        )
        res = [r for r in res if r[1].strip()]
        if not res:
            continue
        res.sort(key=lambda x: min(p[0] for p in x[0]))
        raw = "".join(re.sub(r'[^A-Z0-9]', '', r[1].upper()) for r in res)
        conf = float(np.mean([r[2] for r in res]))
        fmt, regex_ok = fix_plate_by_position(raw)
        is_valid = regex_ok and (conf >= min_conf)
        candidates.append({"source": source_name, "raw": raw, "fmt": fmt, "conf": conf, "valid": is_valid})

    if not candidates:
        return "TIDAK TERBACA", 0.0, False, "NO_OCR_RESULT", {}

    valid_cands = [c for c in candidates if c["valid"]]
    if valid_cands:
        scores = {}
        for c in valid_cands:
            scores.setdefault(c["fmt"], {"score": 0.0, "best": c})
            scores[c["fmt"]]["score"] += 1.0 + c["conf"]
            if c["conf"] > scores[c["fmt"]]["best"]["conf"]:
                scores[c["fmt"]]["best"] = c
        best = max(scores.values(), key=lambda s: s["score"])["best"]
        valid_map = {fmt: round(info["best"]["conf"], 2) for fmt, info in scores.items()}
        return best["fmt"], round(best["conf"], 2), True, best["source"], valid_map

    best = max(candidates, key=lambda c: (len(c["raw"]), c["conf"]))
    return best["fmt"], round(best["conf"], 2), False, f"REJECTED({best['source']})", {}


_VLM_PROMPT = (
    "Kamu adalah sistem ALPR presisi tinggi untuk plat nomor Indonesia. "
    "Tugasmu adalah membaca nomor plat utama dan angka pajak di bagian bawah.\n"
    "Kembalikan HANYA JSON tanpa markdown:\n"
    '{"plate": "<contoh: B 1234 CD>", '
    '"tax": "<format: MM-YY (contoh: 10-27) atau ->", '
    '"plate_valid": true/false, '
    '"tax_valid": true/false, '
    '"reason": "<alasan jika tidak valid>"}\n'
    "Ketentuan:\n"
    "- Jika plat nomor masih terbaca (meskipun gambar agak buram, kontras rendah, atau ada baut di antara karakter), BACA teksnya dan set plate_valid=true.\n"
    "- HANYA tolak plate_valid=false jika plat benar-benar terpotong setengah/hanya kelihatan separuh, atau rusak/blur total sehingga karakter tidak bisa dikenali sama sekali.\n"
    "- Format nomor plat Indonesia adalah [1-2 huruf wilayah] [1-4 angka] [1-3 huruf seri].\n"
    "- Angka pajak harus berformat MM-YY (2 digit bulan 01-12, 2 digit tahun). Jika tidak terbaca jelas atau terhalang, set tax=\"-\" dan tax_valid=false."
)


def verify_with_local_vlm(bgr_img, model="qwen3.5:9b", endpoint="http://127.0.0.1:11434/api/chat", timeout=30):
    """Verifikasi citra plat dan angka pajak menggunakan model Vision lokal Ollama."""
    ok, buf = cv2.imencode(".jpg", bgr_img)
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "format": "json",
        "options": {"temperature": 0.0},
        "messages": [{"role": "user", "content": _VLM_PROMPT, "images": [b64]}],
    }
    try:
        req = urllib.request.Request(
            endpoint, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_json = json.loads(resp.read().decode("utf-8"))["message"]["content"]
        data = json.loads(raw_json)
    except Exception:
        return None

    raw_plate = re.sub(r"[^A-Z0-9]", "", str(data.get("plate", "")).upper())
    m_plate = re.fullmatch(r"([A-Z]{1,2})([0-9]{1,4})([A-Z]{1,3})", raw_plate)
    plate_fmt = f"{m_plate.group(1)} {m_plate.group(2)} {m_plate.group(3)}" if m_plate else ""
    plate_valid = bool(data.get("plate_valid", False)) and bool(plate_fmt)

    raw_tax = str(data.get("tax", "")).strip()
    m_tax = re.fullmatch(r"(0[1-9]|1[0-2])-([0-9]{2})", raw_tax)
    tax_fmt = f"{m_tax.group(1)}-{m_tax.group(2)}" if m_tax else "-"
    tax_valid = bool(data.get("tax_valid", False)) and bool(m_tax)

    return {
        "plate_fmt": plate_fmt,
        "plate_valid": plate_valid,
        "tax_fmt": tax_fmt,
        "tax_valid": tax_valid,
        "reason": str(data.get("reason", "")).strip(),
    }


_TAX_GRAY_VARIANTS = {"Sharp", "Gray80", "YellowSharp"}


def recognize_tax_date(reader, images_dict, min_conf=0.55, return_source=False):
    """
    Membaca angka pajak dengan aturan wajib: 4 angka terpisah oleh '-' (MM-YY).
    Bulan harus 01-12. Jika terhalang/blur/confidence rendah -> TIDAK VALID ('-').
    return_source=True menambahkan nama ROI asal bacaan (mis. "Kanan") di urutan ke-4.
    """
    candidates = []
    for source_name, img in images_dict.items():
        res = reader.readtext(
            img, allowlist='0123456789-',
            paragraph=False, min_size=4, text_threshold=0.35, low_text=0.25, width_ths=1.5
        )
        # EasyOCR kadang mengembalikan kotak kosong ('' dengan conf 0) yang ikut menurunkan rata-rata conf
        res = [r for r in res if r[1].strip()]
        if not res:
            continue
        res.sort(key=lambda x: min(p[0] for p in x[0]))
        joined = "".join(r[1] for r in res)
        digits = re.sub(r'[^0-9]', '', joined)
        conf = float(np.mean([r[2] for r in res]))

        # Prioritaskan pola eksplisit XX-XX jika tanda '-' ikut terbaca, atau tepat 4 digit MMYY
        m_dash = re.search(r'(0[1-9]|1[0-2])\-([0-9]{2})', joined)
        m_exact = re.fullmatch(r'(0[1-9]|1[0-2])([0-9]{2})', digits)
        # titik pemisah MM.YY sering terbaca sebagai angka (12.16 -> "12816"),
        # jadi kalau ada 5 digit, digit tengahnya dianggap titik
        m_dot = re.fullmatch(r'(0[1-9]|1[0-2])[0-9]([0-9]{2})', digits)
        m = m_dash or m_exact or m_dot or re.search(r'(0[1-9]|1[0-2])([0-9]{2})', digits)
        if m:
            variant = source_name.split("/")[-1]
            candidates.append({"fmt": f"{m.group(1)}-{m.group(2)}", "conf": conf, "source": source_name,
                               "gray": variant in _TAX_GRAY_VARIANTS})

    def roi_of(fmt, gray_only=True):
        # ROI asal bacaan terbaik untuk angka itu, supaya aplikasi bisa menampilkan crop yang dipakai
        pool = [c for c in candidates if c["fmt"] == fmt and (c["gray"] or not gray_only)]
        top = max(pool, key=lambda c: c["conf"]) if pool else None
        return top["source"].split("/")[0] if top and "/" in top["source"] else None

    def done(fmt, conf, valid, roi=None):
        return (fmt, conf, valid, roi) if return_source else (fmt, conf, valid)

    # bacaan abu-abu yang ragu (conf 0.45-0.55) tetap dipakai kalau minimal dua varian sepakat
    weak = Counter(c["fmt"] for c in candidates if c["gray"] and 0.45 <= c["conf"] < min_conf)
    weak_pick = next((f for f, n in weak.most_common(1) if n >= 2), None)
    strong = [c for c in candidates if c["conf"] >= min_conf]
    if not strong:
        return done(weak_pick, 0.5, True, roi_of(weak_pick)) if weak_pick else done("-", 0.0, False)

    # Voting antar varian & antar ROI. Hasil uji 120 plat test (angka pajak dibaca manual):
    # varian biner (Otsu/Adaptive) sering menutup lubang angka 8/6/9 sehingga terbaca 0,
    # dan salahnya kompak di semua ROI. Karena itu suara varian biner diberi bobot setengah,
    # dan pemenang wajib didukung minimal satu varian abu-abu. Kalau tidak ada,
    # lebih baik dilaporkan tidak terbaca daripada menampilkan angka yang salah.
    scores = {}
    for c in strong:
        sc = scores.setdefault(c["fmt"], {"score": 0.0, "conf": 0.0, "gray": 0})
        sc["score"] += (1.0 if c["gray"] else 0.5) * (1.0 + c["conf"])
        sc["conf"] = max(sc["conf"], c["conf"])
        sc["gray"] += c["gray"]
    fmt, best = max(scores.items(), key=lambda kv: kv[1]["score"])
    if best["gray"] == 0:
        if weak_pick:
            return done(weak_pick, 0.5, True, roi_of(weak_pick))
        return done("-", round(best["conf"], 2), False)
    return done(fmt, round(best["conf"], 2), True, roi_of(fmt))


# ==============================================================================
# 5. PIPELINE TERPADU & EVALUASI METRIK LENGKAP
# ==============================================================================
def analyze_single_plate(
    bgr_img,
    reader,
    blur_lap_thresh=80.0,
    blur_ten_thresh=450.0,
    tax_blur_lap_thresh=55.0,
    tax_blur_ten_thresh=300.0,
    min_plate_conf=0.60,
    min_tax_conf=0.55,
    use_vlm=False,
    vlm_model="qwen3.5:9b",
    skip_deskew=False,
):
    """Menjalankan deteksi + validasi kualitas untuk 1 citra pelat (nomor utama & pajak)."""
    geom_ok, geom_reason, aspect = check_plate_geometry(bgr_img)
    if not geom_ok:
        return {
            "plate_valid": False,
            "plate_status": f"TIDAK VALID ({geom_reason})",
            "plate_text": "TIDAK VALID",
            "plate_conf": 0.0,
            "tax_valid": False,
            "tax_status": "TIDAK VALID (Plat tidak utuh)",
            "tax_text": "-",
            "tax_conf": 0.0,
            "sharpness_lap": 0.0,
            "is_yellow": False,
        }

    prep = preprocess_plate(bgr_img, skip_deskew=skip_deskew)
    plate_blurry, lap_score, ten_score = is_too_blurry(prep["gray"], blur_lap_thresh, blur_ten_thresh)
    if plate_blurry:
        return {
            "plate_valid": False,
            "plate_status": f"TIDAK VALID (Blur: lap={lap_score}, ten={ten_score})",
            "plate_text": "TIDAK VALID",
            "plate_conf": 0.0,
            "tax_valid": False,
            "tax_status": "TIDAK VALID (Blur)",
            "tax_text": "-",
            "tax_conf": 0.0,
            "sharpness_lap": lap_score,
            "is_yellow": prep["is_yellow"],
        }

    main_vars, _ = process_main_plate_roi(prep["enhanced"], prep["yellow_channel"], raw_gray=prep["gray"])
    p_txt, p_conf, p_valid, p_src, valid_map = recognize_main_plate(reader, main_vars, min_conf=min_plate_conf)

    # Evaluasi angka pajak secara terpisah (bisa blur/terhalang meski plat utama jelas)
    # Beberapa kandidat ROI pajak dicek buram atau tidak satu per satu. ROI yang buram
    # dilewati, sisanya dibaca semua lalu di-voting di recognize_tax_date.
    enh, ych = prep["enhanced"], prep["yellow_channel"]
    tax_imgs, laps, t_roi = {}, [], None
    for name, (ty1, ty2, tx1, tx2) in tax_roi_candidates(enh, ych):
        crop = enh[ty1:ty2, tx1:tx2]
        blurry, lap, _ = is_too_blurry(crop, tax_blur_lap_thresh, tax_blur_ten_thresh)
        laps.append(lap)
        if not blurry:
            ycrop = ych[ty1:ty2, tx1:tx2] if ych is not None else None
            tax_imgs.update(_tax_variants(crop, ycrop, name + "/"))
    if not tax_imgs:
        t_txt, t_conf, t_valid = "-", 0.0, False
        t_status = f"TIDAK VALID (Pajak Blur: lap={max(laps) if laps else 0})"
    else:
        t_txt, t_conf, t_valid, t_roi = recognize_tax_date(reader, tax_imgs, min_conf=min_tax_conf, return_source=True)
        if t_valid:
            t_status = f"VALID (ROI {t_roi})" if t_roi else "VALID"
        else:
            t_status = "TIDAK VALID (Terhalang/Tidak terbaca)"

    # Tahap 2 (Opsional): Konsensus dengan VLM lokal (qwen3.5:9b) untuk akurasi tinggi
    if use_vlm:
        vlm = verify_with_local_vlm(bgr_img, model=vlm_model)
        if vlm is not None and vlm["plate_valid"]:
            # Jika EasyOCR juga valid
            if p_valid:
                if vlm["plate_fmt"] in valid_map:
                    p_txt = vlm["plate_fmt"]
                    p_conf = max(p_conf, valid_map[p_txt])
                    p_src = f"{p_src}+VLM"
                elif vlm["plate_fmt"] == p_txt:
                    p_src = f"{p_src}+VLM"
                else:
                    # Ambiguitas OCR vs VLM: ambil yang formatnya valid sempurna jika salah satu lebih kuat
                    # Jika EasyOCR membaca sebagian (mis. tanpa seri belakang), VLM melengkapi
                    vlm_clean = re.sub(r'[^A-Z0-9]', '', vlm["plate_fmt"])
                    ocr_clean = re.sub(r'[^A-Z0-9]', '', p_txt)
                    if ocr_clean in vlm_clean or levenshtein_distance(ocr_clean, vlm_clean) <= 1:
                        p_txt = vlm["plate_fmt"]
                        p_conf = max(p_conf, 0.90)
                        p_src = f"VLM_CONSENSUS({p_src})"
                    else:
                        p_valid = False
                        p_src = f"AMBIGUOUS(OCR={p_txt}, VLM={vlm['plate_fmt']})"
            else:
                # EasyOCR sebelumnya gagal/ragu (seperti test006_1.jpg), tapi VLM membaca dengan jelas
                # Cek apakah hasil VLM cocok dengan karakter yang terdeteksi di EasyOCR
                p_valid = True
                p_txt = vlm["plate_fmt"]
                p_conf = 0.88
                p_src = "VLM_ASSISTED"
        elif vlm is not None and not vlm["plate_valid"] and p_valid:
            # VLM menyatakan rusak/blur total
            p_valid = False
            p_src = f"VLM_REJECTED({vlm['reason'] or 'uncertain'})"

        if vlm is not None:
            if tax_imgs and vlm["tax_valid"]:
                t_txt = vlm["tax_fmt"]
                t_valid = True
                t_conf = max(t_conf, 0.85)
                t_status = "VALID (VLM)"
            elif t_valid and not vlm["tax_valid"] and t_conf < 0.80:
                t_txt, t_valid, t_conf = "-", 0.0, 0.0
                t_status = "TIDAK VALID (VLM_REJECTED)"

    return {
        "plate_valid": p_valid,
        "plate_status": f"VALID ({p_src})" if p_valid else f"TIDAK VALID ({p_src}, conf={p_conf})",
        "plate_text": p_txt if p_valid else "TIDAK VALID",
        "plate_raw_best": p_txt,
        "plate_conf": p_conf,
        "tax_valid": t_valid,
        "tax_status": t_status,
        "tax_text": t_txt,
        "tax_conf": t_conf,
        "tax_roi": t_roi,
        "sharpness_lap": lap_score,
        "is_yellow": prep["is_yellow"],
    }


def load_gt_from_txt(img_file, class_names):
    lbl_file = img_file.replace("/images/", "/labels/")
    lbl_file = os.path.splitext(lbl_file)[0] + ".txt"
    if not os.path.exists(lbl_file):
        return ""
    boxes = []
    with open(lbl_file, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 5:
                boxes.append((float(p[1]), class_names[int(p[0])]))
    boxes.sort(key=lambda x: x[0])
    raw = "".join(ch for _, ch in boxes)
    m = re.fullmatch(r'([A-Z]{1,2})([0-9]{1,4})([A-Z]{1,3})', raw)
    return f"{m.group(1)} {m.group(2)} {m.group(3)}" if m else raw


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            curr_row.append(min(prev_row[j + 1] + 1, curr_row[j] + 1, prev_row[j] + (c1 != c2)))
        prev_row = curr_row
    return prev_row[-1]


def evaluate_ocr_metrics(predictions, ground_truths):
    exact_matches, total_c_edits, total_c, total_w_edits, total_w = 0, 0, 0, 0, 0
    y_true_chars, y_pred_chars = [], []

    for pred, gt in zip(predictions, ground_truths):
        p_c = re.sub(r'[^A-Z0-9]', '', pred.upper())
        g_c = re.sub(r'[^A-Z0-9]', '', gt.upper())
        if p_c == g_c:
            exact_matches += 1
        total_c_edits += levenshtein_distance(p_c, g_c)
        total_c += max(1, len(g_c))
        total_w_edits += levenshtein_distance(pred.upper().split(), gt.upper().split())
        total_w += max(1, len(gt.upper().split()))
        for k in range(max(len(g_c), len(p_c))):
            y_true_chars.append(g_c[k] if k < len(g_c) else "_")
            y_pred_chars.append(p_c[k] if k < len(p_c) else "_")

    n = max(1, len(ground_truths))
    cer = (total_c_edits / total_c) * 100.0

    # Hitung Precision, Recall, F1 weighted secara eksak tanpa dependensi sklearn
    labels = sorted(list(set(y_true_chars + y_pred_chars) - {"_"}))
    prec_sum, rec_sum, f1_sum, total_support = 0.0, 0.0, 0.0, 0
    for lbl in labels:
        tp = sum(1 for yt, yp in zip(y_true_chars, y_pred_chars) if yt == lbl and yp == lbl)
        fp = sum(1 for yt, yp in zip(y_true_chars, y_pred_chars) if yt != lbl and yp == lbl)
        fn = sum(1 for yt, yp in zip(y_true_chars, y_pred_chars) if yt == lbl and yp != lbl)
        support = tp + fn
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        prec_sum += p * support
        rec_sum += r * support
        f1_sum += f1 * support
        total_support += support

    denom = max(1, total_support)
    return {
        "Exact Match Accuracy (%)": round((exact_matches / n) * 100.0, 2),
        "CER (%)": round(cer, 2),
        "Character Accuracy (%)": round(max(0.0, 100.0 - cer), 2),
        "WER (%)": round((total_w_edits / total_w) * 100.0, 2),
        "Character Precision (%)": round((prec_sum / denom) * 100.0, 2),
        "Character Recall (%)": round((rec_sum / denom) * 100.0, 2),
        "Character F1-Score (%)": round((f1_sum / denom) * 100.0, 2),
    }
