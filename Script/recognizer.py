import re
from datetime import datetime
import cv2
import easyocr
import numpy as np

# Daftar resmi Kode Wilayah TNKB Indonesia (Perpol No. 7 Tahun 2021 + Pemekaran Papua & Khusus)
INDONESIA_REGION_CODES = frozenset({
    # 1 Huruf (Jawa & Banten)
    "A", "B", "D", "E", "F", "G", "H", "K", "L", "M", "N", "P", "R", "S", "T", "W", "Z",
    # 2 Huruf — Sumatera
    "BA", "BB", "BD", "BE", "BG", "BH", "BK", "BL", "BM", "BN", "BP",
    # 2 Huruf — Jawa & DIY
    "AA", "AB", "AD", "AE", "AG",
    # 2 Huruf — Bali & Nusa Tenggara
    "DK", "DR", "EA", "DH", "EB", "ED",
    # 2 Huruf — Kalimantan
    "DA", "KB", "KH", "KT", "KU",
    # 2 Huruf — Sulawesi
    "DB", "DC", "DD", "DL", "DM", "DN", "DP", "DT", "DW",
    # 2 Huruf — Maluku & Papua (termasuk kode lama DS & pemekaran baru PS, PT, PY)
    "DE", "DG", "DS", "PA", "PB", "PS", "PT", "PY",
    # 2 Huruf — Khusus (Pemerintah Pusat & Korps Diplomatik)
    "RI", "CD", "CC",
})

_PREFIX_ALT = "|".join(sorted(INDONESIA_REGION_CODES, key=lambda s: (-len(s), s)))
# Standar TNKB Indonesia: [Kode Wilayah Sah] [Angka 1-9999 tanpa awalan 0] [1-3 Huruf Seri]
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
    "O": "0", "G": "0", "D": "0", "Q": "0", "C": "0", "U": "0",
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
        Cek median piksel tepi/border (lebih tahan pantulan glare dibanding mean global):
        jika latar tepi gelap (< 120), balik secara internal agar teks menjadi gelap di latar terang.
        """
        if image is None or image.size == 0:
            return image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        h, w = gray.shape[:2]
        pad = max(2, min(4, h // 10, w // 10))
        border_pixels = np.concatenate([
            gray[:pad, :].ravel(),
            gray[-pad:, :].ravel(),
            gray[:, :pad].ravel(),
            gray[:, -pad:].ravel(),
        ])
        if float(np.median(border_pixels)) < 120.0:
            return cv2.bitwise_not(image)
        return image

    def parse_and_validate_plate(self, raw_combined):
        """
        Validasi & koreksi posisional terstandar TNKB Indonesia (Perpol No. 7 Tahun 2021):
        [Kode Wilayah Resmi 1-2 Huruf] [Angka 1-9999 tanpa awalan 0] [1-3 Huruf Seri].
        """
        compact = self.clean_text(raw_combined)

        # 1. Cocok langsung terhadap standar penuh TNKB Indonesia
        full_match = TNKB_FULL_REGEX.fullmatch(compact)
        if full_match:
            return f"{full_match.group(1)} {full_match.group(2)} {full_match.group(3)}", True, False

        # 1b. Pelat resmi tanpa huruf seri belakang (0-edit, misal B 1234, RI 1)
        front_match = TNKB_FRONT_REGEX.fullmatch(compact)
        if front_match and len(compact) >= 3:
            return f"{front_match.group(1)} {front_match.group(2)}", True, False

        # 2. Koreksi posisional dengan biaya substitusi minimum (maks 2 karakter)
        if 3 <= len(compact) <= 9:
            best_cand = None
            b_rank = {2: 0, 3: 1, 1: 2}
            for f_len in (1, 2):
                for b_len in (2, 3, 1):
                    m_len = len(compact) - f_len - b_len
                    if not (1 <= m_len <= 4):
                        continue
                    p_raw = compact[:f_len]
                    m_raw = compact[f_len:f_len + m_len]
                    s_raw = compact[f_len + m_len:]

                    p_fix = "".join(DICT_INT_TO_CHAR.get(c, c) for c in p_raw)
                    if p_fix not in INDONESIA_REGION_CODES:
                        continue
                    m_fix = "".join(DICT_CHAR_TO_INT.get(c, c) for c in m_raw)
                    if not re.fullmatch(r"[1-9][0-9]{0,3}", m_fix):
                        continue
                    s_fix = "".join(DICT_INT_TO_CHAR.get(c, c) for c in s_raw)
                    if not re.fullmatch(r"[A-Z]{1,3}", s_fix):
                        continue

                    cand_str = p_fix + m_fix + s_fix
                    edits = sum(a != b for a, b in zip(compact, cand_str))
                    if edits > 2:
                        continue
                    # Prioritas: jumlah substitusi terkecil, lalu preferensi seri 2/3/1 huruf
                    key = (edits, b_rank[b_len], f_len)
                    if best_cand is None or key < best_cand[0]:
                        best_cand = (key, f"{p_fix} {m_fix} {s_fix}")

            if best_cand is not None:
                return best_cand[1], True, False

        # 3. Buang 1 artefak tepi (misal bingkai kiri/kanan terbaca I/1/L/Q) jika sisanya cocok persis TNKB
        if 5 <= len(compact) <= 10:
            for trimmed in (compact[1:], compact[:-1]):
                m_trim = TNKB_FULL_REGEX.fullmatch(trimmed)
                if m_trim:
                    return f"{m_trim.group(1)} {m_trim.group(2)} {m_trim.group(3)}", True, False

        occluded_left = re.match(r"^([1-9][0-9]{0,3})([A-Z]{1,3})$", compact)
        if occluded_left:
            return f"{occluded_left.group(1)} {occluded_left.group(2)}", False, True

        if len(compact) < 4 or compact.isdigit():
            return raw_combined.strip() or "TIDAK TERBACA", False, True

        return raw_combined.strip(), False, False

    def _extract_main_characters(self, image):
        ocr_img = self._prepare_for_ocr(image)
        img_h = ocr_img.shape[0] if ocr_img is not None and ocr_img.size > 0 else 1
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
            y_center = (min(ys) + max(ys)) / 2.0
            max_h = max(max_h, box_h)
            blocks.append({
                "text": clean,
                "conf": float(conf),
                "box_h": box_h,
                "x_left": min(xs),
                "y_center": y_center,
            })

        if not blocks:
            return "", 0.0

        # Saring noise pendek & blok angka pajak yang pusat vertikalnya berada di bawah 72% tinggi citra
        upper_blocks = [
            b for b in blocks
            if b["box_h"] >= 0.40 * max_h
            and not (b["y_center"] > 0.72 * img_h and b["box_h"] < 0.70 * max_h)
        ]
        main_blocks = upper_blocks or [b for b in blocks if b["box_h"] >= 0.40 * max_h] or blocks
        main_blocks.sort(key=lambda item: item["x_left"])
        combined_text = " ".join(b["text"] for b in main_blocks)
        avg_conf = float(np.mean([b["conf"] for b in main_blocks]))
        return combined_text, avg_conf

    def _character_level_consensus(self, valid_cands):
        """
        Voting tingkat karakter untuk kandidat regex-valid yang memiliki panjang karakter sama
        (misal 'H8957FE' vs 'H8957FF'), berbobot nilai confidence.
        """
        groups = {}
        for c in valid_cands:
            compact = c["fmt"].replace(" ", "")
            groups.setdefault(len(compact), []).append((compact, c["conf"]))

        best_char_cand = None
        for length, items in groups.items():
            if len(items) < 2:
                continue
            voted_chars = []
            char_confs = []
            for pos in range(length):
                pos_scores = {}
                for txt, cf in items:
                    pos_scores[txt[pos]] = pos_scores.get(txt[pos], 0.0) + max(0.1, cf)
                best_ch = max(pos_scores.items(), key=lambda kv: kv[1])[0]
                voted_chars.append(best_ch)
                agree_confs = [cf for txt, cf in items if txt[pos] == best_ch]
                char_confs.append(float(np.mean(agree_confs)))

            voted_compact = "".join(voted_chars)
            fmt, ok, occ = self.parse_and_validate_plate(voted_compact)
            if ok:
                avg_c = float(np.mean(char_confs))
                total_w = sum(cf for _, cf in items)
                if best_char_cand is None or total_w > best_char_cand[0]:
                    best_char_cand = (total_w, {
                        "source": "Consensus Voting (Karakter)",
                        "raw": voted_compact,
                        "fmt": fmt,
                        "conf": round(avg_c, 3),
                        "valid": True,
                        "occ": occ,
                        "regex_ok": True,
                    })
        return best_char_cand[1] if best_char_cand else None

    def recognize(self, image_input, fallback_image=None, extra_images=None):
        """Voting OCR multi-kandidat dengan consensus boost & character-level voting."""
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
        fmt_counts = {}
        for name, img in sources:
            raw, conf = self._extract_main_characters(img)
            fmt, val, occ = self.parse_and_validate_plate(raw)
            if fmt and fmt != "TIDAK TERBACA":
                key = fmt.replace(" ", "")
                fmt_counts[key] = fmt_counts.get(key, 0) + 1
                candidates.append({
                    "source": name,
                    "raw": raw,
                    "fmt": fmt,
                    "conf": round(conf, 3),
                    "regex_ok": bool(val),
                    "valid": bool(val and conf >= 0.55),
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

        # Consensus boost: izinkan conf >= 0.35 jika minimal 2 kandidat sepakat pada hasil yang sama
        for c in candidates:
            key = c["fmt"].replace(" ", "")
            if c["regex_ok"] and (c["conf"] >= 0.55 or (c["conf"] >= 0.35 and fmt_counts.get(key, 0) >= 2)):
                c["valid"] = True

        valid_cands = [c for c in candidates if c["valid"]]
        if valid_cands:
            scores = {}
            for c in valid_cands:
                key = c["fmt"].replace(" ", "")
                scores.setdefault(key, {"score": 0.0, "count": 0, "best": c})
                scores[key]["score"] += 1.0 + c["conf"]
                scores[key]["count"] += 1
                if c["conf"] > scores[key]["best"]["conf"]:
                    scores[key]["best"] = c

            best = max(scores.values(), key=lambda s: s["score"])["best"]
            # Jika semua kandidat berbeda-beda (tidak ada yang sepakat >= 2), coba character-level voting
            if max(s["count"] for s in scores.values()) == 1 and len(valid_cands) >= 2:
                char_cand = self._character_level_consensus(valid_cands)
                if char_cand is not None:
                    best = char_cand
        else:
            regex_cands = [c for c in candidates if c["regex_ok"]]
            pool = regex_cands if regex_cands else candidates
            best = max(pool, key=lambda c: (len(c["fmt"].replace(" ", "")), c["conf"]))

        return {
            "raw_text": best["raw"],
            "clean_text": best["fmt"],
            "confidence": round(best["conf"], 2),
            "is_valid_format": best["valid"] or best["regex_ok"],
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
        (memberi bobot prioritas pada varian grayscale dibanding biner agar digit seperti 4 dan 8 tidak terputus)
        dan mengklasifikasikan status PAJAK AKTIF vs MATI / KEDALUWARSA.
        """
        now = current_date or datetime.now()
        inputs = []
        if tax_gray is not None and tax_gray.size > 0:
            inputs.append(("tax_gray", tax_gray, 0.15))
            clahe_tax = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(
                cv2.cvtColor(tax_gray, cv2.COLOR_BGR2GRAY) if tax_gray.ndim == 3 else tax_gray
            )
            inputs.append(("tax_gray_clahe", clahe_tax, 0.12))
        if tax_binary is not None and tax_binary.size > 0:
            inputs.append(("tax_binary", tax_binary, 0.0))

        tax_scores = {}
        for _, im, bonus in inputs:
            ocr_im = self._prepare_for_ocr(im)
            im_h, im_w = ocr_im.shape[:2]
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
            # Abaikan deteksi di 25% ujung kiri (area logo dealer/garis bingkai kiri bawah)
            valid_boxes = [
                r for r in results
                if max(pt[0] for pt in r[0]) >= 0.25 * im_w
            ]
            if not valid_boxes:
                continue
            ordered = sorted(valid_boxes, key=lambda r: min(pt[0] for pt in r[0]))
            raw_joined = "".join(t for _, t, _ in ordered)
            avg_conf = float(np.mean([c for _, _, c in ordered]))
            digits_only = re.sub(r"[^0-9]", "", self._correct_tax_digits(raw_joined))

            m = re.search(r"(0[1-9]|1[0-2])([0-9]{2})", digits_only)
            if m:
                month, year_short = int(m.group(1)), int(m.group(2))
            else:
                # Font ketok angka '0' pada pelat motor putih sering bercelah di kanan atas
                # sehingga terbaca '6' (misal '10.30' terbaca '1636' atau '1630').
                m_open_zero = re.search(r"16([1-3][0-9])", digits_only)
                if not m_open_zero:
                    continue
                month = 10
                year_short = int(m_open_zero.group(1))
                if year_short in (26, 36) and year_short > (now.year % 100) + 5:
                    year_short -= 6

            year_full = 2000 + year_short
            tax_fmt = f"{month:02d}.{year_short:02d}"
            entry = tax_scores.setdefault(
                tax_fmt,
                {"score": 0.0, "best_conf": 0.0, "month": month, "year_full": year_full},
            )
            entry["score"] += 0.5 + avg_conf + bonus
            if avg_conf > entry["best_conf"]:
                entry["best_conf"] = avg_conf

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
        month = best_info["month"]
        year_full = best_info["year_full"]
        tax_conf = round(best_info["best_conf"], 2)
        is_active = (year_full, month) >= (now.year, now.month)
        return {
            "tax_text": best_fmt,
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
