"""
Modul Custom OCR Pelat Nomor Indonesia berbasis YOLOv8 Character Detection (36 Kelas: 0-9, A-Z).
Kompatibel penuh (drop-in replacement) dengan interface PlateRecognizer pada project utama.
"""
import re
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

CLASSES = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
]

INDONESIA_REGION_CODES = frozenset({
    "A", "B", "D", "E", "F", "G", "H", "K", "L", "M", "N", "P", "R", "S", "T", "W", "Z",
    "BA", "BB", "BD", "BE", "BG", "BH", "BK", "BL", "BM", "BN", "BP",
    "AA", "AB", "AD", "AE", "AG",
    "DK", "DR", "EA", "DH", "EB", "ED",
    "DA", "KB", "KH", "KT", "KU",
    "DB", "DC", "DD", "DL", "DM", "DN", "DP", "DT", "DW",
    "DE", "DG", "DS", "PA", "PB", "PS", "PT", "PY",
    "RI", "CD", "CC",
})

_PREFIX_ALT = "|".join(sorted(INDONESIA_REGION_CODES, key=lambda s: (-len(s), s)))
TNKB_FULL_REGEX = re.compile(rf"^({_PREFIX_ALT})([1-9][0-9]{{0,3}})([A-Z]{{1,3}})$")
TNKB_FRONT_REGEX = re.compile(rf"^({_PREFIX_ALT})([1-9][0-9]{{0,3}})$")

DICT_CHAR_TO_INT = {
    "O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "T": "1",
    "Z": "2", "A": "4", "S": "5", "G": "6", "B": "8",
}
DICT_INT_TO_CHAR = {
    "0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B",
}
TAX_DIGIT_CORRECTIONS = {
    "O": "0", "G": "0", "D": "0", "Q": "0", "C": "0",
    "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8",
}


class CustomPlateRecognizer:
    """
    Pembaca Karakter Pelat Nomor berbasis YOLOv8 Character OCR (ocr_best.pt).
    Memiliki method recognize() dan recognize_tax() yang identik dengan PlateRecognizer
    sehingga dapat langsung menggantikan EasyOCR pada aplikasi utama.
    """

    def __init__(self, model_path=None, use_gpu=False, conf_threshold=0.25):
        base_dir = Path(__file__).resolve().parent
        if model_path is None:
            candidates = [
                base_dir / "ocr_best.pt",
                base_dir / "runs" / "detect" / "yolov8n_char_id" / "weights" / "best.pt",
            ]
            model_path = next((p for p in candidates if p.exists()), candidates[0])
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"File bobot model OCR tidak ditemukan di: {self.model_path}\n"
                "Silakan unduh 'ocr_best.pt' hasil training Colab dari Google Drive lalu simpan di folder ini."
            )
        self.device = 0 if use_gpu else "cpu"
        self.conf_threshold = conf_threshold
        self.model = YOLO(str(self.model_path))

    @staticmethod
    def clean_text(text):
        return re.sub(r"[^A-Z0-9]", "", text.upper())

    @staticmethod
    def _ensure_bgr(image):
        if image is None or image.size == 0:
            return None
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image

    def detect_characters(self, image, main_row_only=True):
        """
        Deteksi karakter pada gambar pelat, saring tumpukan kotak (1D NMS),
        dan urutkan dari kiri ke kanan berdasarkan x_center.
        Return: (raw_string, avg_confidence, list_of_char_boxes)
        """
        bgr = self._ensure_bgr(image)
        if bgr is None:
            return "", 0.0, []

        h_img, w_img = bgr.shape[:2]
        preds = self.model.predict(
            bgr,
            imgsz=320,
            conf=self.conf_threshold,
            iou=0.45,
            device=self.device,
            verbose=False,
        )
        if not preds or preds[0].boxes is None or len(preds[0].boxes) == 0:
            return "", 0.0, []

        boxes = preds[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        xywhn = boxes.xywhn.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        raw_items = []
        for (x1, y1, x2, y2), (xc, yc, wn, hn), cid, cf in zip(xyxy, xywhn, cls_ids, confs):
            idx = int(cid)
            ch = CLASSES[idx] if 0 <= idx < len(CLASSES) else "?"
            raw_items.append({
                "char": ch,
                "conf": float(cf),
                "xc": float(xc),
                "yc": float(yc),
                "wn": float(wn),
                "hn": float(hn),
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
            })

        # Filter baris pajak di bagian bawah jika main_row_only=True
        if main_row_only and len(raw_items) >= 3:
            heights = sorted([it["hn"] for it in raw_items], reverse=True)
            ref_h = float(np.median(heights[: min(7, len(heights))]))
            filtered = [
                it for it in raw_items
                if it["yc"] <= 0.75 and it["hn"] >= 0.55 * ref_h
            ]
            if len(filtered) >= 3:
                raw_items = filtered

        # 1D horizontal suppression agar tidak ada 2 huruf di titik x yang sama
        raw_items.sort(key=lambda d: d["conf"], reverse=True)
        kept = []
        for it in raw_items:
            duplicate = any(
                abs(it["xc"] - k["xc"]) < 0.35 * min(it["wn"], k["wn"])
                and abs(it["yc"] - k["yc"]) < 0.45 * min(it["hn"], k["hn"])
                for k in kept
            )
            if not duplicate:
                kept.append(it)

        kept.sort(key=lambda d: d["xc"])
        if not kept:
            return "", 0.0, []

        text = "".join(d["char"] for d in kept)
        avg_conf = float(np.mean([d["conf"] for d in kept]))
        return text, avg_conf, kept

    def parse_and_validate_plate(self, raw_combined):
        """
        Validasi & koreksi posisional sesuai standar TNKB Indonesia (Perpol No. 7 Tahun 2021):
        [Kode Wilayah Resmi 1-2 Huruf] [Angka 1-9999 tanpa awalan 0] [1-3 Huruf Seri].
        """
        compact = self.clean_text(raw_combined)
        if not compact:
            return "TIDAK TERBACA", False, True

        full_match = TNKB_FULL_REGEX.fullmatch(compact)
        if full_match:
            return f"{full_match.group(1)} {full_match.group(2)} {full_match.group(3)}", True, False

        n = len(compact)
        best_full = None
        for p_len in (1, 2):
            for num_len in range(1, 5):
                for s_len in range(1, 4):
                    if p_len + num_len + s_len != n:
                        continue
                    raw_p = compact[:p_len]
                    raw_num = compact[p_len:p_len + num_len]
                    raw_s = compact[p_len + num_len:]

                    edits = 0
                    p_chars = []
                    valid_split = True
                    for ch in raw_p:
                        if ch.isalpha():
                            p_chars.append(ch)
                        elif ch in DICT_INT_TO_CHAR:
                            p_chars.append(DICT_INT_TO_CHAR[ch])
                            edits += 1
                        else:
                            valid_split = False
                            break
                    if not valid_split:
                        continue
                    prefix = "".join(p_chars)
                    if prefix not in INDONESIA_REGION_CODES:
                        continue

                    num_chars = []
                    for ch in raw_num:
                        if ch.isdigit():
                            num_chars.append(ch)
                        elif ch in DICT_CHAR_TO_INT:
                            num_chars.append(DICT_CHAR_TO_INT[ch])
                            edits += 1
                        else:
                            valid_split = False
                            break
                    if not valid_split:
                        continue
                    number = "".join(num_chars)
                    if number[0] == "0":
                        continue

                    s_chars = []
                    for ch in raw_s:
                        if ch.isalpha():
                            s_chars.append(ch)
                        elif ch in DICT_INT_TO_CHAR:
                            s_chars.append(DICT_INT_TO_CHAR[ch])
                            edits += 1
                        else:
                            valid_split = False
                            break
                    if not valid_split:
                        continue
                    suffix = "".join(s_chars)

                    candidate = (edits, -len(prefix), -len(number), f"{prefix} {number} {suffix}")
                    if best_full is None or candidate < best_full:
                        best_full = candidate

        if best_full is not None:
            return best_full[3], True, False

        front_match = TNKB_FRONT_REGEX.fullmatch(compact)
        if front_match:
            return f"{front_match.group(1)} {front_match.group(2)}", True, True

        return compact, False, True

    def recognize(self, image_input, fallback_image=None, extra_images=None):
        """
        Membaca nomor utama pelat dari 1 atau beberapa varian citra (kompatibel penuh dengan PlateRecognizer).
        """
        if image_input is None or (isinstance(image_input, np.ndarray) and image_input.size == 0):
            return {
                "raw_text": "",
                "clean_text": "TIDAK TERBACA",
                "confidence": 0.0,
                "is_valid_format": False,
                "is_occluded": True,
                "source": "-",
                "all_candidates": [],
                "char_boxes": [],
            }

        sources = [("Citra Utama (Custom YOLOv8 OCR)", image_input)]
        if fallback_image is not None:
            sources.append(("Grayscale Tajam", fallback_image))
        if extra_images:
            for idx, im in enumerate(extra_images):
                if im is not None:
                    sources.append((f"Varian #{idx + 1}", im))

        candidates = []
        for name, img in sources:
            for inv_tag, cur_img in (("", img), (" [Invert]", cv2.bitwise_not(img))):
                raw, conf, cboxes = self.detect_characters(cur_img, main_row_only=True)
                if not raw:
                    continue
                fmt, val, occ = self.parse_and_validate_plate(raw)
                if fmt and fmt != "TIDAK TERBACA":
                    candidates.append({
                        "source": f"{name}{inv_tag}",
                        "raw": raw,
                        "fmt": fmt,
                        "conf": round(conf, 3),
                        "regex_ok": bool(val),
                        "valid": bool(val and conf >= 0.40),
                        "occ": occ,
                        "char_boxes": cboxes,
                    })

        if not candidates:
            return {
                "raw_text": "",
                "clean_text": "TIDAK TERBACA",
                "confidence": 0.0,
                "is_valid_format": False,
                "is_occluded": True,
                "source": "-",
                "all_candidates": [],
                "char_boxes": [],
            }

        valid_cands = [c for c in candidates if c["valid"]]
        pool = valid_cands if valid_cands else candidates
        best = max(pool, key=lambda c: (int(c["regex_ok"]), len(c["fmt"].replace(" ", "")), c["conf"]))

        return {
            "raw_text": best["raw"],
            "clean_text": best["fmt"],
            "confidence": round(best["conf"], 2),
            "is_valid_format": best["valid"] or best["regex_ok"],
            "is_occluded": best["occ"],
            "source": best["source"],
            "all_candidates": candidates,
            "char_boxes": best["char_boxes"],
        }

    def recognize_tax(self, tax_binary, tax_gray=None, tax_gray_clahe=None):
        """
        Membaca masa berlaku pajak MM.YY pada ROI pajak menggunakan Custom YOLOv8 OCR.
        """
        now = datetime.now()
        cur_year, cur_month = now.year, now.month

        inputs = []
        if tax_gray is not None and tax_gray.size > 0:
            inputs.append(("tax_gray", tax_gray))
        if tax_gray_clahe is not None and tax_gray_clahe.size > 0:
            inputs.append(("tax_clahe", tax_gray_clahe))
        if tax_binary is not None and tax_binary.size > 0:
            inputs.append(("tax_bin", tax_binary))

        best_pair = None
        for _, img in inputs:
            for cur_img in (img, cv2.bitwise_not(img)):
                raw, conf, _ = self.detect_characters(cur_img, main_row_only=False)
                if not raw:
                    continue
                digits = "".join(
                    c if c.isdigit() else TAX_DIGIT_CORRECTIONS.get(c, "")
                    for c in raw
                )
                for i in range(max(0, len(digits) - 3)):
                    sub = digits[i:i + 4]
                    if len(sub) == 4:
                        mm, yy = int(sub[:2]), int(sub[2:])
                        if 1 <= mm <= 12 and 15 <= yy <= 35:
                            if best_pair is None or conf > best_pair[0]:
                                best_pair = (conf, mm, yy)

        if best_pair is None:
            return {
                "tax_date": "TIDAK TERBACA",
                "tax_status": "TIDAK TERBACA",
                "tax_month": None,
                "tax_year": None,
                "tax_confidence": 0.0,
            }

        conf, mm, yy = best_pair
        full_year = 2000 + yy
        is_active = (full_year > cur_year) or (full_year == cur_year and mm >= cur_month)
        return {
            "tax_date": f"{mm:02d}.{yy:02d}",
            "tax_status": "AKTIF" if is_active else "MATI",
            "tax_month": mm,
            "tax_year": full_year,
            "tax_confidence": round(float(conf), 2),
        }
