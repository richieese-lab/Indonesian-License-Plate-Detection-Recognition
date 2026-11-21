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

    def __init__(self, model_path=None, use_gpu=False, conf_threshold=0.25, hybrid_tax=True):
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
        self.tax_recognizer = None
        if hybrid_tax:
            try:
                from Script.recognizer import PlateRecognizer
                self.tax_recognizer = PlateRecognizer(use_gpu=use_gpu)
            except Exception as exc:
                print(f"[CustomPlateRecognizer] Hybrid tax fallback ke YOLOv8 ({exc})")

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

                    if edits > 2:
                        continue
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

        sources = [("Citra Utama (Custom YOLOv8 OCR)", image_input, True)]
        if fallback_image is not None:
            sources.append(("Grayscale Tajam", fallback_image, True))
        if extra_images:
            for idx, im in enumerate(extra_images):
                if im is not None:
                    sources.append((f"Varian #{idx + 1}", im, False))

        candidates = []
        fmt_counts = {}
        for name, img, allow_invert in sources:
            img_passes = (("", img), (" [Invert]", cv2.bitwise_not(img))) if allow_invert else (("", img),)
            for inv_tag, cur_img in img_passes:
                raw, conf, cboxes = self.detect_characters(cur_img, main_row_only=True)
                if not raw:
                    continue
                fmt, val, occ = self.parse_and_validate_plate(raw)
                if fmt and fmt != "TIDAK TERBACA":
                    key = fmt.replace(" ", "")
                    if val:
                        fmt_counts[key] = fmt_counts.get(key, 0) + 1
                    candidates.append({
                        "source": f"{name}{inv_tag}",
                        "raw": raw,
                        "fmt": fmt,
                        "conf": round(conf, 3),
                        "regex_ok": bool(val),
                        "valid": bool(val and conf >= 0.55),
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

        for c in candidates:
            key = c["fmt"].replace(" ", "")
            if c["regex_ok"] and (c["conf"] >= 0.55 or (c["conf"] >= 0.40 and fmt_counts.get(key, 0) >= 2)):
                c["valid"] = True

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

    @staticmethod
    def _prepare_tax_crop(img):
        """Ambil sisi kanan ROI pajak (buang garis bingkai kiri) dan beri padding agar rasio ~1:3."""
        if img is None or img.size == 0:
            return None
        h, w = img.shape[:2]
        right_crop = img[:, int(0.38 * w):]
        ch, cw = right_crop.shape[:2]
        if ch == 0 or cw == 0:
            return None
        target_h = max(ch, cw // 3)
        pad_top = (target_h - ch) // 2
        pad_bot = target_h - ch - pad_top
        return cv2.copyMakeBorder(right_crop, pad_top, pad_bot, 8, 8, cv2.BORDER_CONSTANT, value=0)

    def recognize_tax(self, tax_binary, tax_gray=None, tax_gray_clahe=None, current_date=None):
        """
        Membaca masa berlaku pajak MM.YY pada ROI pajak.
        Jika mode hybrid aktif, delegasikan ke EasyOCR (PlateRecognizer.recognize_tax) yang lebih
        akurat untuk angka kecil MM.YY; jika tidak, pakai Custom YOLOv8 OCR dengan crop kanan & voting.
        """
        if self.tax_recognizer is not None:
            return self.tax_recognizer.recognize_tax(
                tax_binary,
                tax_gray=tax_gray,
                current_date=current_date,
            )

        now = current_date or datetime.now()
        cur_year, cur_month = now.year, now.month

        inputs = []
        if tax_gray is not None and tax_gray.size > 0:
            inputs.append(("tax_gray", tax_gray, 0.10))
            g = cv2.cvtColor(tax_gray, cv2.COLOR_BGR2GRAY) if tax_gray.ndim == 3 else tax_gray
            clahe_tax = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
            inputs.append(("tax_clahe", clahe_tax, 0.08))
        if tax_gray_clahe is not None and tax_gray_clahe.size > 0:
            inputs.append(("tax_clahe_ext", tax_gray_clahe, 0.08))
        if tax_binary is not None and tax_binary.size > 0:
            inputs.append(("tax_bin", tax_binary, 0.12))

        tax_scores = {}
        for _, img, bonus in inputs:
            cropped = self._prepare_tax_crop(img)
            if cropped is None:
                continue
            raw, conf, _ = self.detect_characters(cropped, main_row_only=False)
            if not raw:
                continue
            digits = "".join(
                c if c.isdigit() else TAX_DIGIT_CORRECTIONS.get(c, "")
                for c in raw
            )
            m = re.search(r"(0[1-9]|1[0-2])([1-4][0-9])", digits)
            if m:
                mm, yy = int(m.group(1)), int(m.group(2))
                fmt = f"{mm:02d}.{yy:02d}"
                entry = tax_scores.setdefault(
                    fmt,
                    {"score": 0.0, "best_conf": 0.0, "month": mm, "year_full": 2000 + yy},
                )
                entry["score"] += 0.5 + float(conf) + bonus
                if conf > entry["best_conf"]:
                    entry["best_conf"] = float(conf)

        if not tax_scores:
            return {
                "tax_text": "TIDAK TERBACA",
                "tax_month": None,
                "tax_year": None,
                "is_tax_active": None,
                "tax_status": "TIDAK TERBACA",
                "tax_confidence": 0.0,
            }

        best_fmt, best_info = max(tax_scores.items(), key=lambda kv: kv[1]["score"])
        mm = best_info["month"]
        full_year = best_info["year_full"]
        is_active = (full_year, mm) >= (cur_year, cur_month)
        return {
            "tax_text": best_fmt,
            "tax_month": mm,
            "tax_year": full_year,
            "is_tax_active": is_active,
            "tax_status": "AKTIF" if is_active else "MATI / KEDALUWARSA",
            "tax_confidence": round(best_info["best_conf"], 2),
        }
