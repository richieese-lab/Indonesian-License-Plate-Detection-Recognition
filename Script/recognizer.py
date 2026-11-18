import re
from datetime import datetime
import cv2
import easyocr
import numpy as np

DICT_CHAR_TO_INT = {
    "O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "T": "1",
    "Z": "2", "A": "4", "S": "5", "G": "6", "B": "8",
}
DICT_INT_TO_CHAR = {
    "0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B",
}
TAX_DIGIT_CORRECTIONS = {
    "O": "0", "G": "0", "D": "0", "Q": "0", "C": "0",
    "I": "1", "L": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
}


class PlateRecognizer:
    """
    Modul Pembaca Karakter Plat & Masa Berlaku Pajak (Tahap 10).
    - Multi-input OCR voting (Biner ROI Utama, Grayscale Tajam, Biner Alternatif).
    - Validasi format plat Indonesia + pembacaan & klasifikasi status pajak MM.YY.
    """

    def __init__(self, use_gpu=False):
        print(f"[PlateRecognizer] Menginisialisasi EasyOCR (GPU={use_gpu})...")
        self.reader = easyocr.Reader(["en"], gpu=use_gpu)

    @staticmethod
    def clean_text(text):
        return re.sub(r"[^A-Z0-9]", "", text.upper())

    @staticmethod
    def _prepare_for_ocr(image):
        """
        EasyOCR paling akurat pada teks gelap di atas latar terang.
        Jika input biner berlatar hitam (0) & teks putih (255), balik secara internal.
        """
        if image is None or image.size == 0:
            return image
        if float(np.mean(image)) < 110.0:
            return cv2.bitwise_not(image)
        return image

    def parse_and_validate_plate(self, raw_combined):
        """
        Validasi & koreksi posisional pola plat nomor Indonesia:
        [1-2 Huruf Wilayah] [1-4 Angka] [1-3 Huruf Seri].
        """
        compact = self.clean_text(raw_combined)

        full_match = re.fullmatch(r"([A-Z]{1,2})([0-9]{1,4})([A-Z]{1,3})", compact)
        if full_match:
            return f"{full_match.group(1)} {full_match.group(2)} {full_match.group(3)}", True, False

        if 4 <= len(compact) <= 9:
            for f_len in (1, 2):
                for b_len in (2, 3, 1):
                    m_len = len(compact) - f_len - b_len
                    if 1 <= m_len <= 4:
                        cand = (
                            "".join(DICT_INT_TO_CHAR.get(c, c) for c in compact[:f_len])
                            + "".join(DICT_CHAR_TO_INT.get(c, c) for c in compact[f_len:f_len + m_len])
                            + "".join(DICT_INT_TO_CHAR.get(c, c) for c in compact[f_len + m_len:])
                        )
                        m2 = re.fullmatch(r"([A-Z]{1,2})([0-9]{1,4})([A-Z]{1,3})", cand)
                        if m2:
                            return f"{m2.group(1)} {m2.group(2)} {m2.group(3)}", True, False

        front_match = re.match(r"^([A-Z]{1,2})([0-9]{1,4})$", compact)
        if front_match and len(compact) >= 3:
            return f"{front_match.group(1)} {front_match.group(2)}", True, False

        occluded_left = re.match(r"^([0-9]{1,4})([A-Z]{1,3})$", compact)
        if occluded_left:
            return f"{occluded_left.group(1)} {occluded_left.group(2)}", False, True

        if len(compact) < 4 or compact.isdigit():
            return raw_combined.strip() or "TIDAK TERBACA", False, True

        return raw_combined.strip(), False, False

    def _extract_main_characters(self, image):
        ocr_img = self._prepare_for_ocr(image)
        results = self.reader.readtext(
            ocr_img,
            allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            paragraph=False,
            min_size=5,
            text_threshold=0.4,
            low_text=0.3,
            width_ths=1.5,
            mag_ratio=1.5,
        )
        if not results:
            return "", 0.0

        blocks = []
        max_h = 0
        for bbox, text, conf in results:
            clean = self.clean_text(text)
            if not clean:
                continue
            ys = [pt[1] for pt in bbox]
            xs = [pt[0] for pt in bbox]
            box_h = max(ys) - min(ys)
            max_h = max(max_h, box_h)
            blocks.append({"text": clean, "conf": float(conf), "box_h": box_h, "x_left": min(xs)})

        if not blocks:
            return "", 0.0

        main_blocks = [b for b in blocks if b["box_h"] >= 0.40 * max_h] or blocks
        main_blocks.sort(key=lambda item: item["x_left"])
        combined_text = " ".join(b["text"] for b in main_blocks)
        avg_conf = float(np.mean([b["conf"] for b in main_blocks]))
        return combined_text, avg_conf

    def recognize(self, image_input, fallback_image=None, extra_images=None):
        """Voting OCR 3 kandidat (Biner Utama, Grayscale Tajam, Biner Alternatif)."""
        if image_input is None or (isinstance(image_input, np.ndarray) and image_input.size == 0):
            return {
                "raw_text": "",
                "clean_text": "TIDAK TERBACA",
                "confidence": 0.0,
                "is_valid_format": False,
                "is_occluded": True,
                "source": "-",
                "all_candidates": [],
            }

        sources = [("Biner ROI Utama (Otsu)", image_input)]
        if fallback_image is not None:
            sources.append(("Grayscale Tajam", fallback_image))
        if extra_images:
            for idx, im in enumerate(extra_images):
                if im is not None:
                    label = "Biner Alternatif (Adaptive)" if idx == 0 else f"Varian #{idx + 1}"
                    sources.append((label, im))

        candidates = []
        for name, img in sources:
            raw, conf = self._extract_main_characters(img)
            fmt, val, occ = self.parse_and_validate_plate(raw)
            if fmt and fmt != "TIDAK TERBACA":
                is_valid = bool(val and conf >= 0.55)
                candidates.append({
                    "source": name,
                    "raw": raw,
                    "fmt": fmt,
                    "conf": round(conf, 3),
                    "valid": is_valid,
                    "occ": occ,
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
            }

        valid_cands = [c for c in candidates if c["valid"]]
        if valid_cands:
            scores = {}
            for c in valid_cands:
                key = c["fmt"].replace(" ", "")
                scores.setdefault(key, {"score": 0.0, "best": c})
                scores[key]["score"] += 1.0 + c["conf"]
                if c["conf"] > scores[key]["best"]["conf"]:
                    scores[key]["best"] = c
            best = max(scores.values(), key=lambda s: s["score"])["best"]
        else:
            best = max(candidates, key=lambda c: (len(c["fmt"].replace(" ", "")), c["conf"]))

        return {
            "raw_text": best["raw"],
            "clean_text": best["fmt"],
            "confidence": round(best["conf"], 2),
            "is_valid_format": best["valid"],
            "is_occluded": best["occ"],
            "source": best["source"],
            "all_candidates": candidates,
        }

    @staticmethod
    def _correct_tax_digits(text):
        cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
        return "".join(TAX_DIGIT_CORRECTIONS.get(ch, ch) for ch in cleaned)

    def recognize_tax(self, tax_binary, tax_gray=None, current_date=None):
        """
        [KEBARUAN 2] Membaca masa berlaku pajak MM.YY pada ROI Pajak
        dan mengklasifikasikan status PAJAK AKTIF vs MATI / KEDALUWARSA.
        """
        now = current_date or datetime.now()
        inputs = []
        if tax_gray is not None and tax_gray.size > 0:
            inputs.append(tax_gray)
        if tax_binary is not None and tax_binary.size > 0:
            inputs.append(tax_binary)

        best_match = None
        for im in inputs:
            ocr_im = self._prepare_for_ocr(im)
            results = self.reader.readtext(
                ocr_im,
                allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ.-",
                paragraph=False,
                min_size=4,
                text_threshold=0.35,
                low_text=0.25,
                mag_ratio=2.0,
            )
            if not results:
                continue
            raw_joined = "".join(t for _, t, _ in results)
            avg_conf = float(np.mean([c for _, _, c in results]))
            digits_only = re.sub(r"[^0-9]", "", self._correct_tax_digits(raw_joined))

            m = re.search(r"(0[1-9]|1[0-2])([1-4][0-9])", digits_only)
            if m:
                month, year_short = int(m.group(1)), int(m.group(2))
                year_full = 2000 + year_short
                cand = (month, year_full, f"{month:02d}.{year_short:02d}", round(avg_conf, 2))
                if best_match is None or cand[3] > best_match[3]:
                    best_match = cand

        if best_match is None:
            return {
                "tax_text": "TIDAK TERBACA",
                "tax_month": None,
                "tax_year": None,
                "is_tax_active": None,
                "tax_status": "TIDAK TERBACA",
                "tax_confidence": 0.0,
            }

        month, year_full, tax_fmt, tax_conf = best_match
        is_active = (year_full, month) >= (now.year, now.month)
        return {
            "tax_text": tax_fmt,
            "tax_month": month,
            "tax_year": year_full,
            "is_tax_active": is_active,
            "tax_status": "AKTIF" if is_active else "MATI / KEDALUWARSA",
            "tax_confidence": tax_conf,
        }


if __name__ == "__main__":
    rec = PlateRecognizer(use_gpu=False)
    for sample in ["T 1187 HB", "A 1330 FP", "5 CSQ", "92 AAY"]:
        fmt, val, occ = rec.parse_and_validate_plate(sample)
        print(f"Plat: '{sample:<10}' -> '{fmt:<10}' | Valid={val} | Terhalang={occ}")
    for tax_sample in ["G1.28", "O1.24", "12.27"]:
        d = rec._correct_tax_digits(tax_sample)
        print(f"Koreksi pajak '{tax_sample}' -> '{d}'")
