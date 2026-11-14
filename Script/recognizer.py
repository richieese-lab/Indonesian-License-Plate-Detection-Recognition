import re
from datetime import datetime
import cv2
import easyocr
import numpy as np


class PlateRecognizer:
    """
    Modul Pembaca Karakter Huruf, Angka Utama, dan Masa Berlaku Pajak (EasyOCR).
    Dilengkapi:
    1. Parameter deteksi sensitif untuk huruf tunggal di kiri/kanan (width_ths=1.5, min_size=5).
    2. Pengurutan teks dari kiri ke kanan (Left-to-Right Spatial Sorting).
    3. Multi-pass OCR (membaca dari 3 kandidat: Biner ROI Utama, Grayscale Tajam, dan Biner Alternatif Adaptive)
       beserta tabel komparasi kandidat (`all_candidates`).
    4. Validasi pola plat nomor Indonesia: [Huruf Wilayah] [Angka] [Huruf Seri].
    5. [KEBARUAN 2] Deteksi & validasi nomor pajak (MM.YY) beserta koreksi huruf-ke-angka
       (misal 'G1.28' -> '01.28') dan klasifikasi status pajak (AKTIF vs MATI / KEDALUWARSA).
    """

    TAX_CHAR_MAP = {
        'O': '0', 'G': '0', 'D': '0', 'Q': '0', 'C': '0', 'U': '0',
        'I': '1', 'L': '1', 'T': '1', '|': '1', '/': '1',
        'Z': '2',
        'E': '3',
        'A': '4',
        'S': '5',
        'B': '8',
    }

    def __init__(self, use_gpu=False):
        print(f"[PlateRecognizer] Menginisialisasi EasyOCR (GPU={use_gpu})...")
        self.reader = easyocr.Reader(['en'], gpu=use_gpu)

    def clean_text(self, text):
        """
        Membersihkan karakter non-alfanumerik.
        """
        return re.sub(r'[^A-Z0-9]', '', text.upper())

    def parse_and_validate_plate(self, raw_combined):
        """
        Mengekstrak pola plat nomor utama Indonesia:
        [1-2 Huruf Wilayah] [1-4 Digit Angka] [1-3 Huruf Seri]
        Contoh: T 1187 HB, A 1330 FP, B 1995 JVK.
        """
        compact = re.sub(r'[^A-Z0-9]', '', raw_combined.upper())

        # 1. Cek pola plat Indonesia lengkap: [1-2 Huruf] + [1-4 Angka] + [1-3 Huruf]
        full_match = re.fullmatch(r'([A-Z]{1,2})([0-9]{1,4})([A-Z]{1,3})', compact)
        if full_match:
            formatted = f"{full_match.group(1)} {full_match.group(2)} {full_match.group(3)}"
            return formatted, True, False

        # 2. Jika ada tambahan noise angka pajak di belakang, cari substring plat utama
        sub_match = re.search(r'([A-Z]{1,2})([0-9]{1,4})([A-Z]{1,3})', compact)
        if sub_match:
            formatted = f"{sub_match.group(1)} {sub_match.group(2)} {sub_match.group(3)}"
            return formatted, True, False

        # 3. Cek pola plat dengan Huruf Wilayah + Angka (tanpa seri belakang)
        front_match = re.match(r'^([A-Z]{1,2})([0-9]{1,4})$', compact)
        if front_match and len(compact) >= 3:
            formatted = f"{front_match.group(1)} {front_match.group(2)}"
            return formatted, True, False

        # 4. Cek indikasi plat TERHALANG di sisi kiri (hanya terbaca Angka + Huruf Seri Belakang)
        occluded_left = re.match(r'^([0-9]{1,4})([A-Z]{1,3})$', compact)
        if occluded_left:
            formatted = f"{occluded_left.group(1)} {occluded_left.group(2)}"
            return formatted, False, True

        # 5. Jika hanya angka saja atau terlalu pendek (< 4 karakter) -> Belum lengkap / Terhalang
        if len(compact) < 4 or compact.isdigit():
            return raw_combined.strip(), False, True

        return raw_combined.strip(), False, False

    def _extract_main_characters(self, image):
        """
        Menjalankan EasyOCR dengan parameter sensitif terhadap huruf tunggal di tepi kiri/kanan.
        Mendukung baik polaritas latar hitam (teks putih) maupun latar putih (teks hitam).
        """
        if image is None or (isinstance(image, np.ndarray) and image.size == 0):
            return "", 0.0

        results = self.reader.readtext(
            image,
            allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            paragraph=False,
            min_size=5,
            text_threshold=0.4,
            low_text=0.3,
            width_ths=1.5,
            mag_ratio=1.5
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
            x_left = min(xs)

            if box_h > max_h:
                max_h = box_h

            blocks.append({
                "text": clean,
                "conf": float(conf),
                "box_h": box_h,
                "x_left": x_left
            })

        if not blocks:
            return "", 0.0

        # Hanya ambil blok teks yang tingginya proporsional dengan baris utama (>= 40% tinggi maks)
        main_blocks = [b for b in blocks if b["box_h"] >= (0.40 * max_h)]
        if not main_blocks:
            main_blocks = blocks

        # Urutkan dari Kiri ke Kanan (Kode Wilayah -> Nomor -> Huruf Seri)
        main_blocks.sort(key=lambda item: item["x_left"])

        combined_text = " ".join([b["text"] for b in main_blocks])
        avg_conf = float(np.mean([b["conf"] for b in main_blocks]))
        return combined_text, avg_conf

    def recognize(self, image_input, fallback_image=None, extra_images=None):
        """
        Multi-kandidat OCR dengan voting (3 kandidat: Biner ROI Utama, Grayscale Tajam, Biner Alternatif Adaptive).
        Mengembalikan pemenang sekaligus list of dict `all_candidates` untuk Tabel Komparasi OCR.
        """
        if image_input is None or (isinstance(image_input, np.ndarray) and image_input.size == 0):
            return {
                "raw_text": "",
                "clean_text": "TIDAK TERBACA",
                "confidence": 0.0,
                "is_valid_format": False,
                "is_occluded": True,
                "source": "-",
                "all_candidates": []
            }

        sources = [("1. Biner ROI Utama (Otsu)", image_input)]
        if fallback_image is not None and isinstance(fallback_image, np.ndarray) and fallback_image.size > 0:
            sources.append(("2. Grayscale Tajam (Unsharp)", fallback_image))
        if extra_images:
            for k, im in enumerate(extra_images):
                if im is not None and isinstance(im, np.ndarray) and im.size > 0:
                    label = "3. Biner Alternatif (Adaptive)" if k == 0 else f"Kandidat Ekstra #{k+1}"
                    sources.append((label, im))

        candidates = []
        all_candidates_table = []

        for name, img in sources:
            raw, conf = self._extract_main_characters(img)
            # Jika citra biner berlatar hitam dan hasil bacaan kosong/kurang valid, cek juga inversinya
            fmt, val, occ = self.parse_and_validate_plate(raw)
            if not val and img.ndim == 2:
                raw_inv, conf_inv = self._extract_main_characters(cv2.bitwise_not(img))
                fmt_inv, val_inv, occ_inv = self.parse_and_validate_plate(raw_inv)
                if (val_inv and not val) or (conf_inv > conf and len(fmt_inv) >= len(fmt)):
                    raw, conf, fmt, val, occ = raw_inv, conf_inv, fmt_inv, val_inv, occ_inv

            entry = {
                "source": name,
                "raw": raw if raw else "-",
                "fmt": fmt if fmt else "TIDAK TERBACA",
                "conf": round(float(conf), 2),
                "valid": bool(val),
                "occ": bool(occ)
            }
            all_candidates_table.append(entry)
            if fmt:
                candidates.append(entry)

        if not candidates:
            return {
                "raw_text": "",
                "clean_text": "TIDAK TERBACA",
                "confidence": 0.0,
                "is_valid_format": False,
                "is_occluded": True,
                "source": "-",
                "all_candidates": all_candidates_table
            }

        valid = [c for c in candidates if c["valid"]]
        if valid:
            scores = {}
            for c in valid:
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
            "all_candidates": all_candidates_table
        }

    # ------------------------------------------------------------------
    # KEBARUAN 2: Deteksi & Klasifikasi Masa Berlaku Pajak (MM.YY)
    # ------------------------------------------------------------------
    def correct_tax_digits(self, raw_str):
        """
        Mengoreksi huruf yang sering salah dibaca dari angka pajak emboss
        (contoh: 'G1.28' -> '01.28', 'O1•28' -> '01.28', '0I.2B' -> '01.28').
        """
        if not raw_str:
            return ""
        upper = raw_str.upper()
        corrected = []
        for ch in upper:
            if ch in self.TAX_CHAR_MAP:
                corrected.append(self.TAX_CHAR_MAP[ch])
            elif ch.isdigit() or ch in ".-•: ":
                corrected.append(ch)
        return "".join(corrected)

    def parse_tax_date(self, raw_str, current_date=None):
        """
        Mengekstrak bulan (MM: 01..12) dan tahun (YY -> 20YY) dari string OCR pajak,
        lalu membandingkannya dengan bulan & tahun saat ini.
        """
        if current_date is None:
            current_date = datetime.now()

        corrected = self.correct_tax_digits(raw_str)
        digits_only = re.sub(r'[^0-9]', '', corrected)

        mm, yy = None, None

        # Pola 1: Ada pemisah titik/spasi/strip (misal '01.28' atau '1.28')
        sep_match = re.search(r'(?<!\d)(0?[1-9]|1[0-2])\s*[\.\-\•\:\s]\s*([0-9]{2})(?!\d)', corrected)
        if sep_match:
            mm = int(sep_match.group(1))
            yy = int(sep_match.group(2))
        else:
            # Pola 2: 4 digit berurutan MMYY (misal '0128', '1227')
            for m in re.finditer(r'(0[1-9]|1[0-2])([1-3][0-9])', digits_only):
                mm = int(m.group(1))
                yy = int(m.group(2))
                break
            # Pola 3: Fallback 4 digit apa saja dengan MM antara 01-12
            if mm is None and len(digits_only) >= 4:
                for i in range(len(digits_only) - 3):
                    cand_m = int(digits_only[i:i+2])
                    cand_y = int(digits_only[i+2:i+4])
                    if 1 <= cand_m <= 12:
                        mm, yy = cand_m, cand_y
                        break

        if mm is None or yy is None:
            return None

        full_year = 2000 + yy
        is_active = (full_year, mm) >= (current_date.year, current_date.month)
        status_str = "AKTIF" if is_active else "MATI / KEDALUWARSA"

        return {
            "tax_text": f"{mm:02d}.{yy:02d}",
            "tax_month": mm,
            "tax_year": full_year,
            "is_tax_active": is_active,
            "tax_status": status_str
        }

    def recognize_tax(self, tax_binary, tax_gray=None, current_date=None):
        """
        Membaca ROI Angka Pajak (MM.YY) menggunakan EasyOCR pada tax_gray dan tax_binary,
        menerapkan koreksi karakter ('G1.28' -> '01.28'), serta mengklasifikasikan
        apakah pajak kendaraan masih AKTIF atau sudah MATI / KEDALUWARSA.
        """
        if current_date is None:
            current_date = datetime.now()

        tax_sources = []
        if tax_gray is not None and isinstance(tax_gray, np.ndarray) and tax_gray.size > 0:
            tax_sources.append(("Tax Grayscale", tax_gray))
            tax_sources.append(("Tax Grayscale Inv", cv2.bitwise_not(tax_gray)))
        if tax_binary is not None and isinstance(tax_binary, np.ndarray) and tax_binary.size > 0:
            tax_sources.append(("Tax Binary", tax_binary))
            tax_sources.append(("Tax Binary Inv", cv2.bitwise_not(tax_binary)))

        best_parsed = None
        best_conf = 0.0
        best_raw = ""

        for _, img in tax_sources:
            # Upscale jika ROI pajak terlalu kecil agar EasyOCR akurat membaca angka kecil
            h, w = img.shape[:2]
            if 0 < h < 90:
                scale = 90.0 / float(h)
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            img_pad = cv2.copyMakeBorder(img, 10, 10, 15, 15, cv2.BORDER_REPLICATE)

            results = self.reader.readtext(
                img_pad,
                allowlist='0123456789OGDQCILTZESBA.- ',
                paragraph=False,
                min_size=4,
                text_threshold=0.3,
                low_text=0.25,
                width_ths=1.8,
                mag_ratio=1.8
            )
            if not results:
                continue

            # Urutkan blok dari kiri ke kanan
            ordered = sorted(results, key=lambda r: min(pt[0] for pt in r[0]))
            combined_raw = " ".join(r[1] for r in ordered)
            avg_conf = float(np.mean([r[2] for r in ordered]))

            parsed = self.parse_tax_date(combined_raw, current_date=current_date)
            if parsed is not None and avg_conf >= best_conf:
                best_parsed = parsed
                best_conf = avg_conf
                best_raw = combined_raw

        if best_parsed is None:
            return {
                "tax_text": "TIDAK TERBACA",
                "tax_raw": best_raw,
                "tax_month": None,
                "tax_year": None,
                "is_tax_active": None,
                "tax_status": "TIDAK TERDETEKSI",
                "tax_confidence": 0.0
            }

        return {
            "tax_text": best_parsed["tax_text"],
            "tax_raw": best_raw,
            "tax_month": best_parsed["tax_month"],
            "tax_year": best_parsed["tax_year"],
            "is_tax_active": best_parsed["is_tax_active"],
            "tax_status": best_parsed["tax_status"],
            "tax_confidence": round(best_conf, 2)
        }


if __name__ == "__main__":
    recognizer = PlateRecognizer(use_gpu=False)
    print("=== Uji Validasi Nomor Utama ===")
    for sample in ["B 1995 JVK", "T 1187 HB", "A 1330 FP", "5 CSQ"]:
        fmt, val, occ = recognizer.parse_and_validate_plate(sample)
        print(f"Input: '{sample:<12}' -> Hasil: '{fmt:<12}' | Valid={val} | Terhalang={occ}")

    print("\n=== Uji Koreksi & Klasifikasi Pajak (Kebaruan 2) ===")
    for tax_sample in ["G1.28", "01•28", "O5 24", "11-27", "0I.2B"]:
        parsed = recognizer.parse_tax_date(tax_sample)
        print(f"Input OCR Pajak: '{tax_sample:<8}' -> Terkoreksi: {parsed}")
