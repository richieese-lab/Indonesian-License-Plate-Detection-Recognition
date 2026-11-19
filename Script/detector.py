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
# TODO: Modul Deteksi Warna Pelat (HSV) -> Jenis Kendaraan
# ======================================================================
class PlateColorClassifier:
    """
    TODO: Implementasikan deteksi warna pelat nomor berbasis ruang warna HSV.
    Target kelas warna:
      - Hitam  : Kendaraan Pribadi (Pelat Hitam)
      - Putih  : Kendaraan Pribadi (Pelat Putih)
      - Kuning : Kendaraan Angkutan Umum / Barang
      - Merah  : Kendaraan Dinas Pemerintah
    """

    def classify(self, deskewed_bgr):
        # TODO: Implementasikan logika deteksi warna pelat (HSV) di sini.
        return {
            "plate_color": "TODO",
            "vehicle_type": "TODO: Klasifikasi Jenis Kendaraan",
            "hsv_summary": "TODO: Deteksi warna pelat belum diimplementasikan",
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
    def deskew_plate(self, crop_bgr):
        """
        Luruskan plat yang miring, maksimal +/- MAX_SKEW_DEG derajat.
        Setelah diputar, area bodi mobil di sekitar plat ikut dipotong.
        Return: (citra_lurus, sudut_koreksi, status)
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return crop_bgr, 0.0, "Citra kosong"

        angle, gain = estimate_skew(crop_bgr)
        if abs(angle) < MIN_SKEW_DEG:
            return crop_bgr.copy(), 0.0, "Lurus (tidak dikoreksi)"
        if gain < MIN_SKEW_GAIN:
            return crop_bgr.copy(), 0.0, f"Tidak dikoreksi (estimasi {angle:+.1f} derajat kurang meyakinkan)"

        rotated = _rotate(crop_bgr, angle)
        out = _tighten_after_rotation(rotated, angle)
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

            processed = self.process_crop(crop_img)
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
