import os
import shutil
import tempfile

import cv2
import numpy as np


# ----------------------------------------------------------------------
# Utilitas data & training
# ----------------------------------------------------------------------
def make_char_yaml(data_root, yaml_path=None):
    """
    Membuat file YAML YOLO dari folder 'Indonesian License Plate Recognition Dataset'.
    data_root harus berisi images/{train,val,test}, labels/{train,val,test}, dan classes.names
    (36 kelas: 0-9 dan A-Z).
    """
    import yaml

    names_file = os.path.join(data_root, "classes.names")
    with open(names_file, "r", encoding="utf-8") as f:
        names = [ln.strip() for ln in f if ln.strip()]

    cfg = {
        "path": data_root,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(names)},
    }
    if yaml_path is None:
        yaml_path = os.path.join(tempfile.gettempdir(), "char_data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    return yaml_path


def train_char_model(data_root, save_path="models/best_char.pt", epochs=60,
                     imgsz=320, batch=64, device=0, base_weights="yolov8s.pt"):
    """
    Melatih YOLO untuk mengenali karakter plat (Materi P21).
    Catatan penting: flip horizontal/vertikal DIMATIKAN karena huruf dan angka akan
    tercermin (mis. 2 vs 5, 6 vs 9) dan merusak pembelajaran.
    """
    from ultralytics import YOLO

    yaml_path = make_char_yaml(data_root)
    model = YOLO(base_weights)
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        name="char_recognizer",
        fliplr=0.0,
        flipud=0.0,
        degrees=5.0,
        mosaic=0.5,
        save=True,
    )
    best = os.path.join(str(results.save_dir), "weights", "best.pt")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    shutil.copy(best, save_path)
    return save_path, yaml_path


# ----------------------------------------------------------------------
# Logika perakitan teks plat dari kotak karakter
# ----------------------------------------------------------------------
def select_main_row(chars):
    """
    Pilih baris utama (huruf/angka besar) dan buang baris angka pajak kecil.
    Pengelompokan dilakukan per baris berdasarkan posisi y, lalu baris dengan total
    tinggi karakter terbesar dianggap baris utama. Tinggi dibandingkan di dalam satu
    baris (relatif), bukan terhadap satu nilai maksimum global.
    """
    if not chars:
        return []

    med_h = float(np.median([c["h"] for c in chars]))
    ordered = sorted(chars, key=lambda c: c["yc"])

    clusters = [[ordered[0]]]
    for c in ordered[1:]:
        mean_yc = np.mean([x["yc"] for x in clusters[-1]])
        if c["yc"] - mean_yc > 0.6 * med_h:
            clusters.append([c])
        else:
            clusters[-1].append(c)

    main = max(clusters, key=lambda cl: sum(x["h"] for x in cl))

    row_med_h = float(np.median([c["h"] for c in main]))
    main = [c for c in main if c["h"] >= 0.5 * row_med_h]
    return sorted(main, key=lambda c: c["xc"])


def format_plate(compact):
    """
    Format plat Indonesia: [1-2 huruf] [1-4 angka] [1-3 huruf].
    Mengembalikan (teks_terformat, valid, terhalang).
    """
    import re

    compact = re.sub(r"[^A-Z0-9]", "", compact.upper())

    m = re.fullmatch(r"([A-Z]{1,2})([0-9]{1,4})([A-Z]{1,3})", compact)
    if m:
        return f"{m.group(1)} {m.group(2)} {m.group(3)}", True, False

    m = re.fullmatch(r"([A-Z]{1,2})([0-9]{1,4})", compact)
    if m and len(compact) >= 3:
        return f"{m.group(1)} {m.group(2)}", True, False

    m = re.fullmatch(r"([0-9]{1,4})([A-Z]{1,3})", compact)
    if m:
        return f"{m.group(1)} {m.group(2)}", False, True

    if len(compact) < 4 or compact.isdigit():
        return compact, False, True

    return compact, False, False


# ----------------------------------------------------------------------
# Pengenal karakter
# ----------------------------------------------------------------------
class CharRecognizer:
    """
    Alternatif Tahap 5: pengenal karakter berbasis YOLO yang dilatih pada
    Indonesian License Plate Recognition Dataset (36 kelas: 0-9, A-Z).
    Bekerja pada potongan plat berwarna (bukan biner), karena data latihnya berwarna.
    """

    def __init__(self, model_path="models/best_char.pt", conf_threshold=0.25, imgsz=320):
        from ultralytics import YOLO

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model karakter tidak ditemukan: {model_path}. "
                "Latih dulu dengan train_char_model()."
            )
        print(f"[CharRecognizer] Memuat model karakter dari: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz

    def recognize(self, plate_img):
        if plate_img is None or plate_img.size == 0:
            return self._empty()

        if plate_img.ndim == 2:
            plate_img = cv2.cvtColor(plate_img, cv2.COLOR_GRAY2BGR)

        results = self.model.predict(
            plate_img,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            iou=0.5,
            agnostic_nms=True,   # satu kotak satu karakter, tanpa duplikat lintas kelas
            verbose=False,
        )

        names = self.model.names
        chars = []
        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
                chars.append({
                    "char": str(names[int(b.cls[0])]),
                    "conf": float(b.conf[0]),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "xc": (x1 + x2) / 2.0,
                    "yc": (y1 + y2) / 2.0,
                    "h": y2 - y1,
                })

        if not chars:
            return self._empty()

        main_row = select_main_row(chars)
        if not main_row:
            return self._empty()

        compact = "".join(c["char"] for c in main_row)
        fmt, valid, occluded = format_plate(compact)
        mean_conf = float(np.mean([c["conf"] for c in main_row]))
        min_conf = float(min(c["conf"] for c in main_row))

        return {
            "raw_text": compact,
            "clean_text": fmt,
            "confidence": round(mean_conf, 2),
            "min_char_confidence": round(min_conf, 2),
            "is_valid_format": valid,
            "is_occluded": occluded,
            "source": "YOLO karakter (Recognition Dataset)",
            "chars": main_row,
            "all_chars": chars,
        }

    @staticmethod
    def _empty():
        return {
            "raw_text": "",
            "clean_text": "TIDAK TERBACA",
            "confidence": 0.0,
            "min_char_confidence": 0.0,
            "is_valid_format": False,
            "is_occluded": True,
            "source": "YOLO karakter (Recognition Dataset)",
            "chars": [],
            "all_chars": [],
        }


def draw_chars(plate_img, result, scale=3):
    """Gambar kotak tiap karakter pada potongan plat (untuk visualisasi di notebook)."""
    img = plate_img if plate_img.ndim == 3 else cv2.cvtColor(plate_img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    main_ids = {id(c) for c in result.get("chars", [])}
    for c in result.get("all_chars", []):
        is_main = id(c) in main_ids
        color = (0, 220, 50) if is_main else (0, 140, 255)
        p1 = (int(c["x1"] * scale), int(c["y1"] * scale))
        p2 = (int(c["x2"] * scale), int(c["y2"] * scale))
        cv2.rectangle(img, p1, p2, color, 2)
        cv2.putText(img, c["char"], (p1[0], max(12, p1[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return img


if __name__ == "__main__":
    # Tes mandiri logika perakitan teks (tanpa model) dengan kotak sintetis.
    def box(ch, xc, yc, h, conf=0.9):
        return {"char": ch, "conf": conf, "xc": xc, "yc": yc, "h": h,
                "x1": xc - 5, "x2": xc + 5, "y1": yc - h / 2, "y2": yc + h / 2}

    sample = [box("L", 20, 40, 60), box("1", 70, 40, 60), box("8", 100, 40, 60),
              box("4", 130, 40, 60), box("6", 160, 40, 60), box("J", 210, 40, 60),
              box("E", 240, 40, 60),
              box("0", 100, 95, 22), box("4", 115, 95, 22), box("2", 140, 95, 22), box("6", 155, 95, 22)]
    row = select_main_row(sample)
    text = "".join(c["char"] for c in row)
    print("Baris utama :", text)
    print("Terformat   :", format_plate(text))
    input("\nTekan Enter untuk menutup jendela ini...")
