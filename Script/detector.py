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
#   plate_color, vehicle_type, hsv_summary, polarity_std, is_inverted,
#   polarity_status, aspect_ratio, is_complete, status
#
# Catatan nilai HSV di OpenCV: H = 0-179 (bukan 0-360), S dan V = 0-255.
# ======================================================================

MAX_SKEW_DEG = 25.0     # kemiringan di atas ini tidak dikoreksi, takut gambarnya malah rusak
MIN_SKEW_DEG = 0.5      # di bawah ini plat dianggap sudah lurus
MIN_SKEW_GAIN = 1.05    # rotasi harus menaikkan skor profil minimal 5%, kalau tidak berarti cuma noise
HOUGH_MIN_SUPPORT = 5.0 # total panjang garis Hough minimal 5x lebar plat baru dipakai sebagai cadangan
EDGE_MARGIN = 3         # kotak yang jaraknya ke tepi gambar <= 3 px dianggap menempel tepi

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


def estimate_skew_hough(plate_bgr, max_angle=MAX_SKEW_DEG):
    """
    Estimasi kedua dari garis lurus terpanjang di plat (bingkai atas/bawah plat).
    Canny dulu untuk dapat tepi (P12), lalu Hough (P17) memvoting garis lurus mana
    yang dilewati piksel-piksel tepi itu. Sudut garis-garis panjang = sudut plat.
    Return: (sudut_koreksi, support) atau (None, 0) kalau tidak ada garis yang cukup panjang.
    support = total panjang garis / lebar plat, makin besar makin kuat buktinya.
    """
    gray = _to_gray(plate_bgr)
    h, w = gray.shape[:2]
    if w < 40 or h < 12:
        return None, 0.0
    scale = 300.0 / w
    gray = cv2.resize(gray, (300, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    w = 300
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # ambang Canny diambil dari median kecerahan, supaya tidak perlu disetel per foto
    med = float(np.median(gray))
    edges = cv2.Canny(gray, int(max(0, 0.66 * med)), int(min(255, 1.33 * med) + 20))

    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=40,
                            minLineLength=int(0.35 * w), maxLineGap=int(0.04 * w))
    if lines is None:
        return None, 0.0

    angles, weights = [], []
    # bentuk array hasil HoughLinesP beda antar versi OpenCV, jadi diratakan dulu ke (N, 4)
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if a > 90:
            a -= 180
        elif a <= -90:
            a += 180
        if abs(a) <= max_angle:
            angles.append(a)
            weights.append(float(np.hypot(x2 - x1, y2 - y1)))
    if not angles:
        return None, 0.0

    # median berbobot panjang garis: garis bingkai yang panjang lebih dipercaya
    order = np.argsort(angles)
    a_sorted = np.array(angles)[order]
    w_sorted = np.array(weights)[order]
    cum = np.cumsum(w_sorted)
    angle = float(a_sorted[np.searchsorted(cum, cum[-1] / 2.0)])
    return angle, float(cum[-1] / w)


def _tighten_after_rotation(img, angle, margin=0.06):
    """
    Kotak YOLO selalu tegak, jadi plat yang miring ikut membawa bodi mobil di
    sudut-sudutnya. Setelah plat diluruskan, ukuran plat asli bisa dihitung balik
    dari ukuran kotak dan sudutnya:
        W = wp*cos + hp*sin
        H = wp*sin + hp*cos
    lalu area di luar plat dipotong supaya pembacaan warna tidak tercampur.
    """
    H, W = img.shape[:2]
    t = np.radians(abs(angle))
    c, s = np.cos(t), np.sin(t)
    d = c * c - s * s
    if d <= 0.2:
        return img
    wp = (c * W - s * H) / d
    hp = (c * H - s * W) / d
    # Cek kewajaran hasil hitungan:
    # - untuk kemiringan <= 25 derajat, lebar plat asli seharusnya masih >= 75% lebar kotak
    # - rasio panjang:tinggi plat hasil hitungan harus masuk akal (plat motor ~2, mobil ~3-4)
    # Kalau tidak wajar, biasanya platnya melengkung atau sudutnya kurang tepat,
    # jadi lebih aman tidak dipotong daripada tulisannya ikut terpotong.
    if wp < 0.75 * W or hp <= 0 or not (1.5 <= wp / hp <= 5.5):
        return img
    wp = min(W, wp * (1 + margin))
    hp = min(H, hp * (1 + 2 * margin))
    x1, y1 = int((W - wp) / 2), int((H - hp) / 2)
    return img[y1:y1 + int(hp), x1:x1 + int(wp)].copy()


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


# ======================================================================
# Tahap 1-3: PlateDetector
# ======================================================================
class PlateDetector:
    """
    Modul Deteksi Plat Nomor Kendaraan (Materi P21: YOLO, P4: Ruang Warna).
    Alur per plat: YOLO -> crop -> deskew -> warna plat (HSV) -> standar polaritas.
    Rasio aspek dipakai untuk menandai plat yang terhalang / tidak utuh.
    """
    def __init__(self, model_path=None, conf_threshold=0.35):
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
    def deskew_plate(self, crop_bgr):
        """
        Luruskan plat yang miring, maksimal +/- MAX_SKEW_DEG derajat.
        Ada dua estimasi sudut:
          1. Integral proyeksi (P17), dari baris-baris tulisan. Ini estimasi utama.
          2. Hough (P12 + P17), dari garis bingkai plat. Dipakai untuk konfirmasi,
             dan jadi cadangan kalau estimasi utama tidak yakin tapi garis bingkainya jelas.
        Hough tidak dijadikan estimasi utama karena di foto asli sering tertipu garis
        bemper, plat motor yang melengkung, dan sudut pandang yang miring (perspektif).
        Setelah diputar, area bodi mobil di sekitar plat ikut dipotong.
        Return: (citra_lurus, sudut_koreksi, status)
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return crop_bgr, 0.0, "Citra kosong"

        angle, gain = estimate_skew(crop_bgr)
        h_angle, support = estimate_skew_hough(crop_bgr)
        profile_ok = abs(angle) >= MIN_SKEW_DEG and gain >= MIN_SKEW_GAIN
        hough_ok = h_angle is not None and support >= HOUGH_MIN_SUPPORT and abs(h_angle) >= MIN_SKEW_DEG

        if profile_ok:
            agree = h_angle is not None and abs(angle - h_angle) <= 2.5
            source = "proyeksi & Hough sepakat" if agree else "integral proyeksi"
        elif hough_ok:
            # estimasi utama ragu, tapi garis bingkai plat panjang dan jelas
            angle, source = h_angle, "Hough, integral proyeksi kurang yakin"
        elif abs(angle) < MIN_SKEW_DEG:
            return crop_bgr.copy(), 0.0, "Lurus (tidak dikoreksi)"
        else:
            return crop_bgr.copy(), 0.0, f"Tidak dikoreksi (estimasi {angle:+.1f} derajat kurang meyakinkan)"

        rotated = _rotate(crop_bgr, angle)
        out = _tighten_after_rotation(rotated, angle)
        status = f"Dikoreksi {angle:+.1f} derajat ({source})"
        if abs(angle) >= MAX_SKEW_DEG - 0.5:
            status += " (mentok batas, cek manual)"
        return out, round(angle, 2), status

    # ------------------------------------------------------------------
    # Tahap 3 (Kebaruan 1): Warna plat -> jenis kendaraan
    # ------------------------------------------------------------------
    def classify_plate_color(self, deskewed_bgr):
        """
        Klasifikasi warna plat dari ruang warna HSV.
        - Merah : hue latar merah, tulisannya putih (saturasi rendah) dan lebih terang dari latar
        - Kuning: hue latar kuning dengan saturasi cukup, tulisannya lebih gelap dari latar
        - Sisanya plat tidak berwarna (S rendah), dibedakan dari terang-gelapnya:
            latar lebih terang dari tulisan -> Putih, sebaliknya -> Hitam
        Ambang-ambangnya diatur dari pengamatan di dataset (split train & val).
        """
        if deskewed_bgr is None or deskewed_bgr.size == 0:
            return {"plate_color": "Hitam", "vehicle_type": VEHICLE_TYPE["Hitam"],
                    "hsv_summary": "Citra kosong"}

        st = _plate_stats(deskewed_bgr)
        text_brighter = st["fg_y"] > st["bg_y"]

        # Kuning yang sangat jelas (hampir semua piksel latar kuning pekat). Di sini urutan
        # terang-gelap tidak dicek, karena pantulan cahaya di plat kuning sering lebih terang
        # dari latarnya sehingga Otsu salah mengira pantulan itu sebagai tulisan.
        yellow_strong = (st["yellow_ratio"] >= 0.90 and st["bg_s"] >= 120 and st["bg_v"] >= 100)

        if (st["red_ratio"] >= 0.30 and st["bg_v"] >= 65 and st["fg_s"] <= 60 and text_brighter):
            color = "Merah"
        elif yellow_strong or (
                st["yellow_ratio"] >= 0.50 and not text_brighter and
                (st["bg_s"] >= 100 or (st["bg_s"] >= 75 and st["bg_v"] < 180) or
                 (st["bg_s"] >= 55 and st["bg_v"] < 100))):
            # plat putih yang kena lampu kekuningan juga punya S sedang, tapi V-nya tinggi.
            # Makanya plat kuning yang S-nya tanggung hanya diterima kalau tidak terlalu terang.
            color = "Kuning"
        else:
            color = "Hitam" if text_brighter else "Putih"

        summary = (f"Latar H={st['bg_h']} S={st['bg_s']} V={st['bg_v']} | "
                   f"Tulisan S={st['fg_s']} V={st['fg_v']} | "
                   f"piksel merah {st['red_ratio']:.0%}, kuning {st['yellow_ratio']:.0%}")
        return {"plate_color": color, "vehicle_type": VEHICLE_TYPE[color], "hsv_summary": summary}

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
    def process_crop(self, crop_bgr):
        """Jalankan tahap 2-3 untuk satu crop plat. Dipakai detect() dan tes mandiri."""
        deskewed, skew_angle, deskew_status = self.deskew_plate(crop_bgr)
        color_info = self.classify_plate_color(deskewed)
        polarity_std, is_inverted, polarity_status = self.standardize_polarity(
            deskewed, color_info["plate_color"])
        return {
            "deskewed": deskewed,
            "skew_angle": skew_angle,
            "deskew_status": deskew_status,
            "plate_color": color_info["plate_color"],
            "vehicle_type": color_info["vehicle_type"],
            "hsv_summary": color_info["hsv_summary"],
            "polarity_std": polarity_std,
            "is_inverted": is_inverted,
            "polarity_status": polarity_status,
        }

    @staticmethod
    def check_aspect_ratio(w, h):
        """
        Rasio panjang:tinggi plat Indonesia sekitar 2.0 - 4.5.
        Terlalu kecil (< 1.8) biasanya platnya terpotong / terhalang,
        terlalu besar (> 5.5) biasanya salah deteksi.
        """
        aspect_ratio = float(w) / float(h)
        if aspect_ratio < 1.8:
            return aspect_ratio, False, "Terhalang / Tidak Utuh (Rasio Terlalu Kecil)"
        if aspect_ratio > 5.5:
            return aspect_ratio, False, "Tidak Wajar (Rasio Terlalu Panjang)"
        return aspect_ratio, True, "Lengkap"

    @staticmethod
    def touches_image_edge(box, img_w, img_h, margin=EDGE_MARGIN):
        """
        YOLO cuma bisa mengotaki bagian plat yang kelihatan. Kalau plat keluar dari foto,
        kotaknya akan mentok di tepi gambar. Titik (0, 0) ada di pojok kiri atas, jadi:
          kiri  -> x1 dekat 0        kanan -> x2 dekat lebar gambar
          atas  -> y1 dekat 0        bawah -> y2 dekat tinggi gambar
        Cek rasio aspek saja tidak cukup, karena plat mobil yang terpotong separuh
        rasionya bisa mirip plat motor yang utuh.
        """
        x1, y1, x2, y2 = box
        return x1 <= margin or y1 <= margin or x2 >= img_w - margin or y2 >= img_h - margin

    # ------------------------------------------------------------------
    # Tahap 1: Deteksi & crop
    # ------------------------------------------------------------------
    def detect(self, image_input):
        """
        Mendeteksi plat nomor dari gambar.
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
        results = self.model(image, conf=self.conf_threshold, verbose=False)

        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())

                # jaga supaya koordinat tidak keluar dari ukuran gambar
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_w, x2), min(img_h, y2)
                w, h = x2 - x1, y2 - y1
                if w <= 10 or h <= 10:
                    continue

                aspect_ratio, is_complete, status_desc = self.check_aspect_ratio(w, h)
                if self.touches_image_edge((x1, y1, x2, y2), img_w, img_h):
                    is_complete = False
                    status_desc = "Terpotong di Tepi Gambar"
                crop_img = image[y1:y2, x1:x2]

                det = {
                    "box": (x1, y1, x2, y2),
                    "confidence": conf,
                    "class_id": cls_id,
                    "crop": crop_img,
                }
                det.update(self.process_crop(crop_img))
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
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "Dataset", "Raw", "Indonesian License Plate Dataset")
    out_dir = os.path.join(base_dir, "Output", "01_cropped_plates")
    os.makedirs(out_dir, exist_ok=True)

    test_imgs = sys.argv[1:] or sorted(glob.glob(os.path.join(data_dir, "images", "test", "*.jpg")))[:5]
    if not test_imgs:
        print("Tidak ada gambar uji. Jalankan: python Script/detector.py path/ke/gambar.jpg")
        sys.exit(0)

    detector = PlateDetector()

    for img_path in test_imgs:
        name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(data_dir, "labelswithLP", "test", name + ".txt")
        print(f"\n=== {img_path}")

        if detector.is_plate_model:
            items = [(d["box"], d, None) for d in detector.detect(img_path)]
        elif os.path.exists(label_path):
            # best.pt belum ada: kotak plat diambil dari label dataset, tahap 2-3 tetap bisa diuji
            print("    (mode label: kotak plat diambil dari file label, bukan dari YOLO)")
            items = []
            for box, crop, gt in _crops_from_labels(img_path, label_path):
                d = {"box": box, "crop": crop}
                d.update(detector.process_crop(crop))
                items.append((box, d, gt))
        else:
            print("    Lewati: best.pt belum ada dan gambar ini tidak punya file label.")
            continue

        print(f"    Jumlah plat: {len(items)}")
        for i, (box, d, gt) in enumerate(items, start=1):
            print(f"    Plat #{i}{(' (' + gt + ')') if gt else ''} box={box}")
            print(f"      Deskew   : {d['deskew_status']}")
            print(f"      Warna    : {d['plate_color']} -> {d['vehicle_type']}")
            print(f"      HSV      : {d['hsv_summary']}")
            print(f"      Polaritas: {d['polarity_status']}")
            for key in ("crop", "deskewed", "polarity_std"):
                cv2.imwrite(os.path.join(out_dir, f"{name}_{i}_{key}.jpg"), d[key])

    print(f"\nHasil crop, deskewed, dan polarity_std disimpan di: {out_dir}")
