import os
import sys
import glob
import cv2
import numpy as np
from ultralytics import YOLO

# ======================================================================
# BLOK HULU (Zulfan): YOLO -> crop -> deskew -> warna plat -> polaritas
#
# Key yang dikembalikan untuk setiap plat (sesuai kontrak data tim):
#   box, confidence, class_id, crop, deskewed, skew_angle, deskew_status,
#   trim_box, trim_status,
#   plate_color, vehicle_type, hsv_summary, polarity_std, is_inverted,
#   polarity_status, aspect_ratio, is_complete, status
#
# Catatan nilai HSV di OpenCV: H = 0-179 (bukan 0-360), S dan V = 0-255.
# ======================================================================

MAX_SKEW_DEG = 45.0     # kemiringan di atas ini tidak dikoreksi, takut gambarnya malah rusak
MIN_SKEW_DEG = 0.5      # di bawah ini plat dianggap sudah lurus
MIN_SKEW_GAIN = 1.05    # rotasi harus menaikkan skor profil minimal 5%, kalau tidak berarti cuma noise

VEHICLE_TYPE = {
    "Putih":  "Kendaraan Pribadi (Pelat Putih)",
    "Hitam":  "Kendaraan Pribadi (Pelat Hitam)",
    "Kuning": "Kendaraan Angkutan Umum / Barang",
    "Merah":  "Kendaraan Dinas Pemerintah",
}


# ----------------------------------------------------------------------
# Fungsi bantu
# ----------------------------------------------------------------------
def _to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _rotate(img, angle, border=cv2.BORDER_REPLICATE):
    """Putar citra sebesar `angle` derajat terhadap titik tengahnya (positif = berlawanan jarum jam)."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=border)


def _rotate_expand(img, angle, border=cv2.BORDER_REPLICATE):
    """
    Putar citra tanpa memotong isinya: kanvas diperbesar mengikuti ukuran hasil putaran.
    Kotak YOLO plat miring memuat plat dari sudut ke sudut, jadi kalau diputar di kanvas
    yang sama ukurannya, ujung-ujung plat (huruf terakhir, angka pajak) ikut terbuang.
    """
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(round(w * c + h * s)), int(round(w * s + h * c))
    M[0, 2] += nw / 2.0 - w / 2.0
    M[1, 2] += nh / 2.0 - h / 2.0
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_CUBIC, borderMode=border)


def _rotate_with_context(context, box_xywh, angle):
    """
    Seperti _rotate_expand, tapi bahan putarannya potongan gambar asli yang lebih besar
    dari kotak YOLO. Jadi bagian yang "masuk" ke kanvas saat diputar adalah isi foto
    sebenarnya, bukan piksel tepi yang direntangkan (BORDER_REPLICATE membuat garis-garis
    panjang palsu yang bisa dikira garis bingkai oleh trim_plate_margins).
    Pusat putaran = pusat kotak YOLO, dan hasilnya selalu menaruh pusat itu di tengah kanvas.
    """
    x, y, w, h = box_xywh
    cx, cy = x + w / 2.0, y + h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(round(w * c + h * s)), int(round(w * s + h * c))
    M[0, 2] += nw / 2.0 - cx
    M[1, 2] += nh / 2.0 - cy
    return cv2.warpAffine(context, M, (nw, nh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _text_mask(gray):
    """Mask tulisan (255). Hasil Otsu dibalik kalau perlu, karena piksel tulisan selalu lebih sedikit dari latar."""
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.count_nonzero(b) > b.size / 2:
        b = cv2.bitwise_not(b)
    return b


def estimate_skew(plate_bgr, max_angle=MAX_SKEW_DEG):
    """
    Cari sudut koreksi pakai profil proyeksi horizontal.
    Idenya: kalau plat sudah lurus, piksel tulisan menumpuk di baris yang sama,
    jadi variansi jumlah piksel per baris paling besar. Coba banyak sudut, ambil
    yang variansinya tertinggi.
    Return: (sudut_koreksi, gain). gain = skor sudut terbaik / skor tanpa rotasi.
    """
    gray = _to_gray(plate_bgr)
    h, w = gray.shape[:2]
    if w > 200:
        gray = cv2.resize(gray, (200, max(1, int(h * 200 / w))), interpolation=cv2.INTER_AREA)
    mask = _text_mask(gray)
    mh, mw = mask.shape[:2]
    mask = mask[int(mh * .05):int(mh * .95), int(mw * .03):int(mw * .97)]
    if mask.size == 0 or mask.shape[0] < 8 or mask.shape[1] < 16:
        return 0.0, 1.0

    def score(a):
        r = _rotate(mask, a, border=cv2.BORDER_CONSTANT)
        return float(np.var(r.sum(axis=1, dtype=np.float64)))

    base = score(0.0)
    coarse = np.arange(-max_angle, max_angle + 0.1, 1.0)
    best = max(coarse, key=score)
    fine = np.arange(best - 1.0, best + 1.01, 0.25)
    best = max(fine, key=score)
    gain = score(best) / base if base > 0 else 1.0
    return float(np.clip(best, -max_angle, max_angle)), gain


def _tighten_after_rotation(img, angle, box_w, box_h, margin=0.06):
    """
    Kotak YOLO selalu tegak, jadi plat yang miring ikut membawa bodi mobil di
    sudut-sudutnya. Ukuran plat asli bisa dihitung balik dari ukuran kotak dan sudutnya:
        W = wp*cos + hp*sin
        H = wp*sin + hp*cos
    `img` adalah hasil _rotate_expand, jadi plat ada tepat di tengah kanvas. Bagian
    tengah seukuran plat itulah yang diambil.
    """
    H, W = img.shape[:2]
    t = np.radians(abs(angle))
    c, s = np.cos(t), np.sin(t)
    d = c * c - s * s
    # kalau hitungan tidak bisa dipakai, ambil tengah seukuran kotak asli (sama seperti
    # rotasi biasa), sisa bodi mobil nanti dibuang trim_plate_margins()
    fallback_w, fallback_h = min(W, box_w), min(H, box_h)
    wp = hp = None
    if d > 0.2:
        wp = (c * box_w - s * box_h) / d
        hp = (c * box_h - s * box_w) / d
        # Cek kewajaran: rasio panjang:tinggi plat harus masuk akal (plat motor ~2, mobil ~3-4).
        # Pada sudut besar hp adalah selisih dua angka yang mirip, jadi salah sudut 1-2 derajat
        # sudah membuatnya meleset jauh. Kalau tidak wajar, pakai cadangan.
        if hp <= 0 or wp <= 0 or not (1.5 <= wp / hp <= 5.5):
            wp = hp = None
    if wp is None:
        wp, hp = fallback_w, fallback_h
        m_w, m_h = 0.0, 0.0
    else:
        m_w, m_h = margin, 2 * margin
    wp = min(W, wp * (1 + m_w))
    hp = min(H, hp * (1 + m_h))
    x1, y1 = int(round((W - wp) / 2)), int(round((H - hp) / 2))
    return img[y1:y1 + int(round(hp)), x1:x1 + int(round(wp))].copy()


def _plate_stats(plate_bgr):
    """
    Pisahkan latar dan tulisan pakai Otsu di grayscale (kelompok piksel terbanyak = latar),
    lalu ukur warnanya. Fungsi ini deterministik, jadi klasifikasi warna dan polaritas
    yang sama-sama memanggilnya pasti dapat hasil yang konsisten.
    """
    h, w = plate_bgr.shape[:2]
    inner = plate_bgr[int(h * .15):int(h * .85), int(w * .05):int(w * .95)]
    if inner.size == 0:
        inner = plate_bgr
    gray = _to_gray(inner)
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = m > 0
    bg = bright if bright.mean() >= 0.5 else ~bright
    fg = ~bg
    if not fg.any():
        fg = bg

    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    Hh, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hb, sb, vb = Hh[bg], S[bg], V[bg]

    # piksel latar yang terlalu gelap hue-nya tidak bisa dipercaya, jadi tidak ikut dihitung
    ok = vb >= 60
    hs, ss = (hb[ok], sb[ok]) if ok.any() else (hb, sb)
    red_ratio = float(np.mean(((hs <= 10) | (hs >= 160)) & (ss >= 70))) if hs.size else 0.0
    yellow_ratio = float(np.mean((hs >= 12) & (hs <= 40) & (ss >= 50))) if hs.size else 0.0

    return {
        "bg_h": int(np.median(hb)), "bg_s": int(np.median(sb)), "bg_v": int(np.median(vb)),
        "fg_s": int(np.median(S[fg])), "fg_v": int(np.median(V[fg])),
        # luminansi (nilai grayscale) dipakai untuk polaritas, karena tahap berikutnya kerja di grayscale
        "bg_y": float(gray[bg].mean()), "fg_y": float(gray[fg].mean()),
        "red_ratio": red_ratio, "yellow_ratio": yellow_ratio,
    }


# ----------------------------------------------------------------------
# Pemangkasan pinggiran non-plat (setelah deskew)
# ----------------------------------------------------------------------
TRIM_ROW_THR = 0.30     # baris dengan latar plat < 30% dianggap garis bingkai / bukan plat
TRIM_COL_THR = 0.22     # sama untuk kolom (lebih rendah karena goresan huruf tegak juga mengurangi latar)


def _plate_bg_mask(gray):
    """
    Mask latar plat. Ambang Otsu dihitung dari bagian tengah plat saja (pasti isinya plat),
    lalu kelompok yang paling banyak di tengah itu dianggap warna latar.
    Return: (mask_latar, latar_terang)
    """
    h, w = gray.shape
    seed = gray[int(h * .25):int(h * .75), int(w * .2):int(w * .8)]
    t, sb = cv2.threshold(seed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright_bg = (sb > 0).mean() >= 0.5
    return ((gray > t) if bright_bg else (gray <= t)), bright_bg


def _main_text_box(text_mask):
    """
    Kotak gabungan huruf nomor utama dari connected component tulisan.
    Dipakai sebagai jangkar: pemangkasan tidak boleh melewati kotak ini.
    Return (x1, y1, x2, y2, tinggi_huruf) atau None kalau hurufnya tidak jelas.
    """
    h, w = text_mask.shape
    n, _, st, _ = cv2.connectedComponentsWithStats(text_mask.astype(np.uint8), 8)
    # batas bawah tinggi huruf sengaja longgar (18% tinggi crop): kalau kotak YOLO longgar,
    # plat cuma menempati sebagian crop sehingga hurufnya terlihat pendek. Stiker pajak yang
    # berbentuk kotak penuh disaring lewat kepadatan piksel dan bentuknya yang melebar.
    chars = [tuple(st[i][:4]) for i in range(1, n)
             if 0.18 * h <= st[i][3] <= 0.85 * h and 0.02 * w <= st[i][2] <= 0.22 * w
             and 0.15 <= st[i][4] / float(st[i][2] * st[i][3]) <= 0.85
             and st[i][3] >= 0.9 * st[i][2]]
    if len(chars) < 3:
        return None
    # huruf nomor utama tingginya hampir sama, jadi buang yang jauh dari median tinggi
    mh = float(np.median([c[3] for c in chars]))
    chars = [c for c in chars if 0.7 * mh <= c[3] <= 1.3 * mh]
    if len(chars) < 3:
        return None
    x1 = min(c[0] for c in chars)
    x2 = max(c[0] + c[2] for c in chars)
    y1 = int(np.median([c[1] for c in chars]))
    y2 = int(np.median([c[1] + c[3] for c in chars]))
    return x1, y1, x2, y2, mh


def _scan_until_border(profile, start, step, thr, run=1):
    """Jalan dari `start` ke arah `step` sampai ketemu `run` indeks berturut-turut di bawah `thr`."""
    i, cnt = start, 0
    while 0 <= i < len(profile):
        cnt = cnt + 1 if profile[i] < thr else 0
        if cnt >= run:
            return i - step * (run - 1)
        i += step
    return None


def _ink_mask(gray, bright_bg):
    """
    Mask tulisan/garis dengan ambang adaptif (rata-rata lokal). Garis bingkai plat sering
    tipis dan agak kabur, jadi dengan Otsu global sebagian pikselnya ikut masuk latar.
    """
    blk = max(11, (gray.shape[0] // 3) | 1)
    if bright_bg:
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blk, 5)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blk, -5)


def _long_lines(ink, length, horizontal):
    """
    Opening morfologi (P13-14) dengan elemen struktur berupa garis panjang: semua yang lebih
    pendek dari `length` (huruf, angka, noise) hilang, yang tersisa hanya garis bingkai.
    Sebelumnya mask ditebalkan sedikit supaya garis yang agak miring tidak putus.
    """
    if horizontal:
        pre = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(length)), 1))
    else:
        pre = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, int(length))))
    return cv2.morphologyEx(pre, cv2.MORPH_OPEN, k)


def _has_char_between(ink, ty1, ty2, xa, xb, char_h, tax_y2=None):
    """Cek apakah di antara garis tegak dan kotak huruf masih ada huruf utama atau angka pajak."""
    if xb - xa < 2:
        return False
    n, _, st, _ = cv2.connectedComponentsWithStats((ink[ty1:ty2, xa:xb] > 0).astype(np.uint8), 8)
    if any(st[i][3] >= 0.5 * char_h and st[i][4] >= 0.1 * char_h * char_h for i in range(1, n)):
        return True
    if tax_y2 is not None and tax_y2 > ty2:
        n2, _, st2, _ = cv2.connectedComponentsWithStats((ink[ty2:tax_y2, xa:xb] > 0).astype(np.uint8), 8)
        if any(st2[i][3] >= 0.35 * char_h and st2[i][4] >= 0.05 * char_h * char_h for i in range(1, n2)):
            return True
    return False


def _tax_content_bounds(text_mask, tx1, ty2, tx2, ch):
    """
    Cari batas kanan & bawah angka pajak (MM • YY) di bawah nomor utama agar pemangkasan
    tepi tidak memotong digit tahun pajak yang sering sejajar atau lebih ke kanan dari tx2.
    """
    h, w = text_mask.shape
    tax_end = min(h - 1, int(ty2 + 0.65 * ch))
    tax_bot_lim = min(h, int(ty2 + 1.35 * ch))
    if tax_bot_lim <= ty2:
        return tx2, tax_end
    n, _, st, _ = cv2.connectedComponentsWithStats(text_mask.astype(np.uint8), 8)
    min_x = tx1 + int(0.35 * (tx2 - tx1))
    min_h = max(5.0, 0.35 * ch)
    min_area = max(8.0, 0.05 * ch * ch)
    tax_x2, tax_y2 = tx2, tax_end
    for i in range(1, n):
        sx, sy, sw, sh, area = int(st[i][0]), int(st[i][1]), int(st[i][2]), int(st[i][3]), int(st[i][4])
        if sy < max(0, ty2 - 2) or (sy + sh) > tax_bot_lim:
            continue
        if not (min_h <= sh <= 0.80 * ch and 2 <= sw <= 0.22 * w and sh >= 0.6 * sw):
            continue
        if sx < min_x or (sx + sw) >= int(0.95 * w) or area < min_area:
            continue
        if not (0.15 <= area / float(sw * sh) <= 0.88):
            continue
        tax_x2 = max(tax_x2, sx + sw)
        tax_y2 = max(tax_y2, sy + sh)
    return tax_x2, min(h - 1, tax_y2)


def trim_plate_margins(plate_bgr):
    """
    Buang pinggiran yang bukan plat (bingkai dudukan plat, bodi mobil) dari plat yang sudah lurus.

    Cara kerja (integral proyeksi, P17):
      1. Pisahkan latar plat dan tulisan (Otsu dari bagian tengah plat).
      2. Cari kotak huruf nomor utama serta batas angka pajak (MM • YY) di bawahnya.
      3. Dari tepi kotak huruf/pajak, telusuri profil baris ke atas dan ke bawah, dan profil kolom
         ke kiri dan ke kanan. Begitu ketemu baris/kolom yang hampir tidak berisi latar plat
         (garis bingkai), di situlah tepi plat.
         Penelusuran dari dalam ke luar, bukan dari luar ke dalam, karena bodi mobil di luar
         bingkai bisa saja warnanya sama dengan latar plat (mobil putih, plat putih).
      4. Ke bawah dan ke kanan, titik awalnya memperhitungkan baris angka pajak supaya digit
         terakhir pajak tidak ikut terpotong, disertai ruang aman beberapa piksel di luar teks.
      5. Garis bingkai tipis dicari juga dengan opening morfologi (P13-14), karena di profil
         latar garis setipis itu kurang kelihatan.
    Return: (citra_terpangkas, (y1, y2, x1, x2), status)
    """
    gray = _to_gray(plate_bgr)
    h, w = gray.shape[:2]
    if h < 16 or w < 40:
        return plate_bgr, (0, h, 0, w), "Tidak dipangkas (plat terlalu kecil)"

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    bg, bright_bg = _plate_bg_mask(blur)
    box = _main_text_box(~bg)
    if box is None:
        return plate_bgr, (0, h, 0, w), "Tidak dipangkas (huruf tidak terdeteksi)"
    tx1, ty1, tx2, ty2, ch = box
    # Nomor utama selalu membentang lebih dari separuh lebar plat. Kalau kotak hurufnya
    # sempit, berarti sebagian besar huruf menyatu dengan bingkai (biasanya plat kecil/jauh)
    # dan yang terdeteksi cuma beberapa potongan. Memangkas dari kotak sesempit itu malah
    # memotong huruf, jadi lebih aman dibiarkan.
    if (tx2 - tx1) < 0.5 * w:
        return plate_bgr, (0, h, 0, w), "Tidak dipangkas (huruf hanya terdeteksi sebagian)"
    tax_x2, tax_end = _tax_content_bounds(~bg, tx1, ty2, tx2, ch)
    bgf = bg.astype(np.float32)

    rows = bgf[:, tx1:tx2].mean(axis=1)
    top = _scan_until_border(rows, ty1 - max(1, int(0.08 * ch)), -1, TRIM_ROW_THR)
    bottom = _scan_until_border(rows, tax_end, +1, TRIM_ROW_THR)
    y1 = 0 if top is None else top + 1
    y2 = h if bottom is None else bottom

    # profil kolom diambil di seluruh tinggi plat: garis bingkai tegak memotong semua baris,
    # sedangkan goresan huruf hanya setinggi huruf, jadi latar di kolom huruf masih sekitar 40%
    cols = bgf[y1:y2, :].mean(axis=0)
    mx = max(1, int(0.15 * ch))
    left = _scan_until_border(cols, tx1 - mx, -1, TRIM_COL_THR, run=2)
    right = _scan_until_border(cols, tax_x2 + mx, +1, TRIM_COL_THR, run=2)
    x1 = 0 if left is None else left + 1
    x2 = w if right is None else right

    # Garis bingkai yang tipis sering lolos dari profil latar di atas (pikselnya separuh
    # masuk latar). Garis ini dicari terpisah dengan opening morfologi, lalu batas potong
    # digeser ke garis terdalam yang masih di luar teks.
    ink = _ink_mask(blur, bright_bg)
    hl = _long_lines(ink, 0.45 * (tx2 - tx1), horizontal=True)
    vl = _long_lines(ink, 0.90 * ch, horizontal=False)
    row_line = (hl[:, tx1:tx2] > 0).mean(axis=1)
    col_line = (vl[y1:y2, :] > 0).mean(axis=0)
    above = [y for y in range(0, max(0, ty1 - 1)) if row_line[y] > 0.3]
    below = [y for y in range(min(h, tax_end + 1), h) if row_line[y] > 0.3]
    if above:
        y1 = max(y1, max(above) + 1)
    if below:
        y2 = min(y2, min(below))
    # garis tegak hanya dipakai kalau di antara garis itu dan kotak huruf/pajak tidak ada huruf lagi
    left_limit = min(max(0, tx1 - 1), int(0.14 * w))
    for x in sorted((x for x in range(0, left_limit) if col_line[x] > 0.25), reverse=True):
        if not _has_char_between(ink, ty1, ty2, x + 1, tx1 - 1, ch, tax_y2=tax_end):
            x1 = max(x1, x + 1)
            break
    right_start = max(tax_x2 + 1, int(0.86 * w))
    for x in sorted(x for x in range(right_start, w) if col_line[x] > 0.25):
        if not _has_char_between(ink, ty1, ty2, tax_x2 + 1, x, ch, tax_y2=tax_end):
            x2 = min(x2, x)
            break

    # Jaga ruang aman di luar huruf utama & baris pajak agar efek dilasi garis atau kemiringan
    # perspektif tidak memotong sisi kanan angka tahun pajak (mis. digit '5' pada 12•25).
    pad_x = max(2, int(0.16 * ch))
    pad_y = max(2, int(0.14 * ch))
    x1 = min(x1, max(0, tx1 - pad_x))
    x2 = max(x2, min(w, tax_x2 + pad_x))
    y1 = min(y1, max(0, ty1 - pad_y))
    y2 = max(y2, min(h, tax_end + pad_y))

    # batas aman: pinggiran non-plat tidak pernah selebar ini, kalau lebih berarti salah baca
    x1, x2 = min(x1, int(0.15 * w)), max(x2, int(0.85 * w))
    y1, y2 = min(y1, int(0.30 * h)), max(y2, int(0.70 * h))
    if (x1, y1, x2, y2) == (0, 0, w, h):
        return plate_bgr, (0, h, 0, w), "Tidak dipangkas (tidak ada bingkai di luar plat)"
    status = f"Dipangkas atas {y1}, bawah {h - y2}, kiri {x1}, kanan {w - x2} px"
    return plate_bgr[y1:y2, x1:x2].copy(), (y1, y2, x1, x2), status


# ======================================================================
# Modul Deteksi Warna Pelat (HSV) -> Jenis Kendaraan (Kebaruan 1)
# ======================================================================
class PlateColorClassifier:
    """
    Deteksi warna pelat nomor berbasis ruang warna HSV (Materi P4).
    Target kelas warna:
      - Hitam  : Kendaraan Pribadi (Pelat Hitam)
      - Putih  : Kendaraan Pribadi (Pelat Putih)
      - Kuning : Kendaraan Angkutan Umum / Barang
      - Merah  : Kendaraan Dinas Pemerintah
    """

    def classify(self, deskewed_bgr):
        if deskewed_bgr is None or deskewed_bgr.size == 0:
            return {
                "plate_color": "Hitam",
                "vehicle_type": VEHICLE_TYPE["Hitam"],
                "hsv_summary": "Citra kosong",
            }

        st = _plate_stats(deskewed_bgr)
        text_brighter = st["fg_y"] > st["bg_y"]

        yellow_strong = (st["yellow_ratio"] >= 0.85 and st["bg_s"] >= 110 and st["bg_v"] >= 95)
        red_plate = (
            st["red_ratio"] >= 0.25
            and st["bg_s"] >= 60
            and st["bg_v"] >= 55
            and (text_brighter or st["fg_v"] >= st["bg_v"] - 10)
        )
        yellow_plate = yellow_strong or (
            st["yellow_ratio"] >= 0.45
            and not text_brighter
            and (
                st["bg_s"] >= 95
                or (st["bg_s"] >= 70 and st["bg_v"] < 185)
                or (st["bg_s"] >= 55 and st["bg_v"] < 110)
            )
        )

        if red_plate:
            color = "Merah"
        elif yellow_plate:
            color = "Kuning"
        elif abs(st["bg_y"] - st["fg_y"]) >= 12.0:
            color = "Hitam" if text_brighter else "Putih"
        else:
            color = "Putih" if st["bg_v"] >= 125 else "Hitam"

        summary = (
            f"Latar H={st['bg_h']} S={st['bg_s']} V={st['bg_v']} | "
            f"Merah {st['red_ratio']:.0%}, Kuning {st['yellow_ratio']:.0%}"
        )
        return {
            "plate_color": color,
            "vehicle_type": VEHICLE_TYPE[color],
            "hsv_summary": summary,
        }


# ======================================================================
# Tahap 1-3: PlateDetector
# ======================================================================
class PlateDetector:
    """
    Modul Deteksi Plat Nomor Kendaraan (Materi P21: YOLO, P4: Ruang Warna).
    Alur per plat: YOLO -> crop -> deskew -> warna plat (HSV) -> standar polaritas.
    Rasio aspek dipakai untuk menandai plat yang terhalang / tidak utuh.
    """
    def __init__(self, model_path=None, conf_threshold=None):
        # None = mode otomatis (kalibrasi ambang adaptif di detect())
        self.conf_threshold = conf_threshold

        # Urutan pencarian bobot: models/best.pt -> models/license_plate_detector.pt -> yolov8n.pt
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_model = os.path.join(base_dir, "models", "best.pt")
        alt_model = os.path.join(base_dir, "models", "license_plate_detector.pt")

        if model_path and os.path.exists(model_path):
            self.model_path = model_path
        elif os.path.exists(default_model):
            self.model_path = default_model
        elif os.path.exists(alt_model):
            self.model_path = alt_model
        else:
            # yolov8n bawaan akan di-download otomatis oleh Ultralytics kalau belum ada
            self.model_path = "yolov8n.pt"

        print(f"[PlateDetector] Memuat model YOLO dari: {self.model_path}")
        self.model = YOLO(self.model_path)

        # yolov8n bawaan dilatih di COCO (80 kelas: mobil, orang, dst.), tidak punya kelas plat
        names = getattr(self.model, "names", {}) or {}
        self.is_plate_model = 0 < len(names) <= 2
        if not self.is_plate_model:
            print("[PlateDetector] PERINGATAN: model ini bukan model plat nomor (best.pt belum ada). "
                  "Kotak yang terdeteksi kemungkinan besar bukan plat.")

    # ------------------------------------------------------------------
    # Tahap 2: Deskewing
    # ------------------------------------------------------------------
    def deskew_plate(self, crop_bgr, context=None):
        """
        Luruskan plat yang miring, maksimal +/- MAX_SKEW_DEG derajat.
        Setelah diputar, area bodi mobil di sekitar plat ikut dipotong.
        context: (gambar_lebih_besar, (x, y)) yaitu potongan foto di sekitar kotak YOLO dan
                 posisi kiri-atas crop di dalamnya. Kalau ada, rotasi memakai isi foto asli.
        Return: (citra_lurus, sudut_koreksi, status)
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return crop_bgr, 0.0, "Citra kosong"

        angle, gain = estimate_skew(crop_bgr)
        if abs(angle) < MIN_SKEW_DEG:
            return crop_bgr.copy(), 0.0, "Lurus (tidak dikoreksi)"
        if gain < MIN_SKEW_GAIN:
            return crop_bgr.copy(), 0.0, f"Tidak dikoreksi (estimasi {angle:+.1f} derajat kurang meyakinkan)"

        if context is not None:
            ctx_img, (ox, oy) = context
            rotated = _rotate_with_context(ctx_img, (ox, oy, crop_bgr.shape[1], crop_bgr.shape[0]), angle)
        else:
            rotated = _rotate_expand(crop_bgr, angle)
        out = _tighten_after_rotation(rotated, angle, crop_bgr.shape[1], crop_bgr.shape[0])
        status = f"Dikoreksi {angle:+.1f} derajat"
        if abs(angle) >= MAX_SKEW_DEG - 0.5:
            status += " (mentok batas, cek manual)"
        return out, round(angle, 2), status

    # ------------------------------------------------------------------
    # Tahap 3 (Kebaruan 1): Warna plat -> jenis kendaraan
    # ------------------------------------------------------------------
    def classify_plate_color(self, deskewed_bgr):
        """
        TODO: Implementasikan deteksi warna pelat pada kelas PlateColorClassifier.
        """
        # TODO: Panggil implementasi deteksi warna pelat dari PlateColorClassifier
        return PlateColorClassifier().classify(deskewed_bgr)

    # ------------------------------------------------------------------
    # Tahap 3 (lanjutan): Standarisasi polaritas
    # ------------------------------------------------------------------
    def standardize_polarity(self, deskewed_bgr, plate_color=None):
        """
        Standar tim: LATAR GELAP, TULISAN TERANG.
        - Plat Kuning dan Merah: langsung ikut warnanya. Kuning (tulisan hitam) di-invert,
          Merah (tulisan putih) dibiarkan. Warna ini sudah dipastikan lewat hue, jadi lebih
          bisa dipercaya daripada perbandingan terang-gelap yang gampang terganggu pantulan.
        - Plat Putih/Hitam atau warna tidak diketahui: bandingkan luminansi (grayscale) latar
          dan tulisan. Sengaja tidak pakai V dari HSV, karena V = max(R,G,B) sehingga latar
          merah punya V tinggi padahal di grayscale sebenarnya gelap.
        Return: (citra_polaritas_standar, is_inverted, status)
        """
        if deskewed_bgr is None or deskewed_bgr.size == 0:
            return deskewed_bgr, False, "Citra kosong"

        st = _plate_stats(deskewed_bgr)
        diff = st["bg_y"] - st["fg_y"]
        if plate_color in ("Kuning", "Merah"):
            inverted = plate_color == "Kuning"
            reason = f"plat {plate_color}, tulisan {'gelap' if inverted else 'terang'}"
        elif abs(diff) < 15 and plate_color in VEHICLE_TYPE:
            inverted = plate_color == "Putih"
            reason = f"kontras tipis, ikut warna plat {plate_color}"
        else:
            inverted = diff > 0
            reason = "latar lebih terang dari tulisan" if inverted else "latar sudah lebih gelap dari tulisan"

        out = cv2.bitwise_not(deskewed_bgr) if inverted else deskewed_bgr.copy()
        status = ("Di-invert" if inverted else "Tidak di-invert") + f" ({reason})"
        return out, inverted, status

    # ------------------------------------------------------------------
    def process_crop(self, crop_bgr, context=None):
        """Jalankan tahap 2-3 untuk satu crop plat. Dipakai detect() dan tes mandiri."""
        deskewed, skew_angle, deskew_status = self.deskew_plate(crop_bgr, context)
        # setelah lurus, pinggiran bingkai/bodi mobil dibuang supaya tahap berikutnya
        # (warna, polaritas, binerisasi) hanya melihat plat
        deskewed, trim_box, trim_status = trim_plate_margins(deskewed)
        color_info = self.classify_plate_color(deskewed)
        polarity_std, is_inverted, polarity_status = self.standardize_polarity(
            deskewed, color_info["plate_color"])
        return {
            "deskewed": deskewed,
            "skew_angle": skew_angle,
            "deskew_status": deskew_status,
            "trim_box": trim_box,
            "trim_status": trim_status,
            "plate_color": color_info["plate_color"],
            "vehicle_type": color_info["vehicle_type"],
            "hsv_summary": color_info["hsv_summary"],
            "polarity_std": polarity_std,
            "is_inverted": is_inverted,
            "polarity_status": polarity_status,
        }

    @staticmethod
    def check_aspect_ratio(w, h, box=None, img_shape=None):
        """
        Rasio panjang:tinggi plat Indonesia sekitar 2.0 - 4.5 (mobil) dan 1.45 - 2.5 (motor dari sudut miring).
        Terlalu kecil (< 1.38) atau terpotong di tepi gambar berarti plat terhalang / tidak utuh,
        terlalu besar (> 5.5) biasanya salah deteksi.
        """
        aspect_ratio = float(w) / float(h)
        if aspect_ratio < 1.38:
            return aspect_ratio, False, "Terhalang / Tidak Utuh (Rasio Terlalu Kecil)"
        if aspect_ratio > 5.5:
            return aspect_ratio, False, "Tidak Wajar (Rasio Terlalu Panjang)"
        if box is not None and img_shape is not None:
            x1, y1, x2, y2 = box
            img_h, img_w = img_shape[:2]
            touches_edge = (x1 <= 2 or y1 <= 2 or x2 >= img_w - 2 or y2 >= img_h - 2)
            if touches_edge and aspect_ratio < 1.75:
                return aspect_ratio, False, "Terhalang / Terpotong Tepi Gambar"
        return aspect_ratio, True, "Lengkap"

    @staticmethod
    def _has_plate_texture(crop_bgr):
        """
        Verifikasi ringan untuk kandidat berkeyakinan sedang: memastikan crop memiliki
        kepadatan tepi/kontras karakter pelat, bukan area polos (bodi/aspal/bayangan).
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return False
        gray = _to_gray(crop_bgr)
        h, w = gray.shape[:2]
        if h < 10 or w < 14:
            return False
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        edge_energy = float(np.mean(np.abs(gx)))
        std_val = float(np.std(gray))
        return edge_energy >= 12.0 and std_val >= 22.0

    def _run_yolo_pass(self, image, infer_conf, infer_imgsz):
        results = self.model(
            image,
            conf=infer_conf,
            iou=0.45,
            imgsz=infer_imgsz,
            verbose=False,
        )
        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                boxes.append((x1, y1, x2, y2, conf, cls_id))
        return boxes

    def _detect_tilted_fallback(self, image, infer_conf, infer_imgsz):
        """Fallback untuk foto dengan pelat miring ekstrem (30-45 derajat)."""
        img_h, img_w = image.shape[:2]
        center = (img_w / 2.0, img_h / 2.0)
        mapped = []
        for ang in (35.0, -35.0):
            M = cv2.getRotationMatrix2D(center, ang, 1.0)
            rot = cv2.warpAffine(image, M, (img_w, img_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            raw_boxes = self._run_yolo_pass(rot, max(0.20, infer_conf), infer_imgsz)
            if not raw_boxes:
                continue
            M_inv = cv2.invertAffineTransform(M)
            for rx1, ry1, rx2, ry2, conf, cls_id in raw_boxes:
                corners = np.array([[rx1, ry1, 1], [rx2, ry1, 1], [rx2, ry2, 1], [rx1, ry2, 1]], dtype=np.float32)
                orig_pts = (M_inv @ corners.T).T
                ox1, oy1 = int(np.min(orig_pts[:, 0])), int(np.min(orig_pts[:, 1]))
                ox2, oy2 = int(np.max(orig_pts[:, 0])), int(np.max(orig_pts[:, 1]))
                mapped.append((ox1, oy1, ox2, oy2, conf, cls_id))
            if mapped:
                break
        return mapped

    # ------------------------------------------------------------------
    # Tahap 1: Deteksi & crop
    # ------------------------------------------------------------------
    def detect(self, image_input):
        """
        Mendeteksi plat nomor dari gambar dengan ambang deteksi otomatis.
        Input : path file (str) atau numpy.ndarray BGR
        Output: list of dict, key-nya lihat daftar di bagian atas file.
        """
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Gagal membaca gambar dari: {image_input}")
        else:
            image = image_input.copy()

        img_h, img_w = image.shape[:2]
        auto_mode = self.conf_threshold is None
        infer_conf = 0.18 if auto_mode else float(self.conf_threshold)
        if self.is_plate_model:
            infer_imgsz = 640 if max(img_h, img_w) <= 720 else 960
        else:
            infer_imgsz = 640

        raw_boxes = self._run_yolo_pass(image, infer_conf, infer_imgsz)
        if not raw_boxes and self.is_plate_model:
            raw_boxes = self._detect_tilted_fallback(image, infer_conf, infer_imgsz)

        detections = []
        for x1, y1, x2, y2, conf, cls_id in raw_boxes:
            # jaga supaya koordinat tidak keluar dari ukuran gambar
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            w, h = x2 - x1, y2 - y1
            if w <= 14 or h <= 10:
                continue

            raw_ar = float(w) / float(h)
            # Pelat yang miring 30-45 derajat memiliki bounding box hampir persegi (ar ~ 0.65 - 1.35)
            if raw_ar < 0.65 or raw_ar > 6.0:
                continue

            crop_img = image[y1:y2, x1:x2]

            # Filter otomatis untuk kandidat berkeyakinan sedang (0.18 - 0.32):
            # hanya lolos jika rasio wajar & memiliki tekstur tepi karakter pelat.
            if auto_mode and conf < 0.32:
                if not (0.70 <= raw_ar <= 5.0 and self._has_plate_texture(crop_img)):
                    continue

            # potongan foto yang lebih besar dari kotak, dipakai saat plat diputar
            px, py = w // 2, h // 2
            cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
            context = (image[cy1:min(img_h, y2 + py), cx1:min(img_w, x2 + px)], (x1 - cx1, y1 - cy1))
            processed = self.process_crop(crop_img, context)
            if auto_mode and conf < 0.35 and w >= 75 and "huruf tidak terdeteksi" in processed.get("trim_status", "").lower():
                continue
            # Hitung aspek rasio: untuk pelat miring (>= 15 deg), kotak axis-aligned tampak persegi,
            # jadi evaluasi keutuhan mempertimbangkan kemiringan agar pelat miring utuh tidak dianggap terhalang.
            dw, dh = processed["deskewed"].shape[1], processed["deskewed"].shape[0]
            if abs(processed["skew_angle"]) >= 5.0 and dw > 0 and dh > 0:
                check_w, check_h = dw, dh
            else:
                check_w, check_h = w, h

            aspect_ratio, is_complete, status_desc = self.check_aspect_ratio(
                check_w, check_h, box=(x1, y1, x2, y2), img_shape=(img_h, img_w)
            )
            if not is_complete and abs(processed["skew_angle"]) >= 15.0:
                touches_edge = (x1 <= 2 or y1 <= 2 or x2 >= img_w - 2 or y2 >= img_h - 2)
                if not touches_edge and raw_ar >= 0.65:
                    is_complete = True
                    status_desc = f"Lengkap (Miring {processed['skew_angle']:+.1f} deg)"

            det = {
                "box": (x1, y1, x2, y2),
                "confidence": conf,
                "class_id": cls_id,
                "crop": crop_img,
            }
            det.update(processed)
            det.update({
                "aspect_ratio": round(aspect_ratio, 2),
                "is_complete": is_complete,
                "status": status_desc,
            })
            detections.append(det)

        return detections


# ======================================================================
# Tes mandiri:  python Script/detector.py [path_gambar ...]
# ======================================================================
def _crops_from_labels(img_path, label_path):
    """Ambil crop plat dari file label YOLO (dipakai kalau best.pt belum ada)."""
    img = cv2.imread(img_path)
    H, W = img.shape[:2]
    crops = []
    for line in open(label_path, encoding="utf-8"):
        p = line.split()
        if len(p) < 5:
            continue
        cx, cy, bw, bh = map(float, p[1:5])
        x1, y1 = max(0, int((cx - bw / 2) * W)), max(0, int((cy - bh / 2) * H))
        x2, y2 = min(W, int((cx + bw / 2) * W)), min(H, int((cy + bh / 2) * H))
        if x2 - x1 > 10 and y2 - y1 > 10:
            crops.append(((x1, y1, x2, y2), img[y1:y2, x1:x2], " ".join(p[5:])))
    return crops


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Penggunaan: python Script/detector.py <path_gambar ...>")
        sys.exit(0)

    detector = PlateDetector()
    for img_path in sys.argv[1:]:
        print(f"\n=== {img_path}")
        items = detector.detect(img_path)
        print(f"    Jumlah plat: {len(items)}")
        for i, d in enumerate(items, start=1):
            print(f"    Plat #{i} box={d['box']} conf={d['confidence']:.2f} | {d['status']}")
            print(f"      Deskew   : {d['deskew_status']}")
            print(f"      Warna    : {d['plate_color']} -> {d['vehicle_type']}")
            print(f"      Polaritas: {d['polarity_status']}")
