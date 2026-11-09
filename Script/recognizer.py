import re
import cv2
import easyocr
import numpy as np

class PlateRecognizer:
    """
    Modul Pembaca Karakter Huruf dan Angka (EasyOCR).
    Dilengkapi:
    1. Parameter deteksi sensitif untuk huruf tunggal di kiri/kanan (width_ths=1.5, min_size=5).
    2. Pengurutan teks dari kiri ke kanan (Left-to-Right Spatial Sorting).
    3. Multi-pass OCR (membaca dari citra biner bersih & citra grayscale upscaled).
    4. Validasi pola plat nomor Indonesia: [Huruf Wilayah] [Angka] [Huruf Seri].
    """
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
        Contoh: T 1187 HB, A 1330 FP, B 1234 ABC.
        """
        compact = re.sub(r'[^A-Z0-9]', '', raw_combined.upper())

        # 1. Cek pola plat Indonesia lengkap: [1-2 Huruf] + [1-4 Angka] + [1-3 Huruf]
        full_match = re.fullmatch(r'([A-Z]{1,2})([0-9]{1,4})([A-Z]{1,3})', compact)
        if full_match:
            formatted = f"{full_match.group(1)} {full_match.group(2)} {full_match.group(3)}"
            return formatted, True, False

        # 2. Cek pola plat dengan Huruf Wilayah + Angka (tanpa seri belakang)
        front_match = re.match(r'^([A-Z]{1,2})([0-9]{1,4})$', compact)
        if front_match and len(compact) >= 3:
            formatted = f"{front_match.group(1)} {front_match.group(2)}"
            return formatted, True, False

        # 3. Cek indikasi plat TERHALANG di sisi kiri (hanya terbaca Angka + Huruf Seri Belakang)
        #    Contoh: "5 CSQ" atau "92 AAY"
        occluded_left = re.match(r'^([0-9]{1,4})([A-Z]{1,3})$', compact)
        if occluded_left:
            formatted = f"{occluded_left.group(1)} {occluded_left.group(2)}"
            return formatted, False, True

        # 4. Jika hanya angka saja atau terlalu pendek (< 4 karakter) -> Belum lengkap / Terhalang
        if len(compact) < 4 or compact.isdigit():
            return raw_combined.strip(), False, True

        return raw_combined.strip(), False, False

    def _extract_main_characters(self, image):
        """
        Menjalankan EasyOCR dengan parameter sensitif terhadap huruf tunggal di tepi kiri/kanan
        (seperti huruf 'T', 'A', 'B', 'D', 'N', 'L', 'W').
        """
        results = self.reader.readtext(
            image,
            allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            paragraph=False,
            min_size=5,          # Agar huruf tunggal (A, T, L, B) tidak diabaikan
            text_threshold=0.4,  # Lebih sensitif menangkap huruf di tepi
            low_text=0.3,
            width_ths=1.5,       # Menggabungkan spasi lebar antara Huruf - Angka - Huruf
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
                "conf": conf,
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
        Multi-kandidat OCR dengan voting.
        Membaca citra biner utama, citra fallback (grayscale tajam), dan varian tambahan
        (mis. biner Adaptive). Hasil dipilih: format plat valid dengan suara terbanyak
        (dijumlah confidence), jika tidak ada yang valid pilih yang paling panjang/yakin.
        """
        if image_input is None or (isinstance(image_input, np.ndarray) and image_input.size == 0):
            return {
                "raw_text": "",
                "clean_text": "TIDAK TERBACA",
                "confidence": 0.0,
                "is_valid_format": False,
                "is_occluded": True
            }

        sources = [("Biner utama (Otsu)", image_input)]
        if fallback_image is not None:
            sources.append(("Grayscale tajam", fallback_image))
        if extra_images:
            for k, im in enumerate(extra_images):
                if im is not None:
                    sources.append((f"Biner alternatif #{k+1} (Adaptive)", im))

        candidates = []
        for name, img in sources:
            raw, conf = self._extract_main_characters(img)
            fmt, val, occ = self.parse_and_validate_plate(raw)
            if fmt:
                candidates.append({"raw": raw, "conf": conf, "fmt": fmt, "valid": val, "occ": occ, "source": name})

        if not candidates:
            return {
                "raw_text": "",
                "clean_text": "TIDAK TERBACA",
                "confidence": 0.0,
                "is_valid_format": False,
                "is_occluded": True
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
            "all_candidates": [(c["source"], c["fmt"], round(c["conf"], 2)) for c in candidates]
        }

if __name__ == "__main__":
    recognizer = PlateRecognizer(use_gpu=False)
    for sample in ["T 1187 HB", "A 1330 FP", "5 CSQ", "92 AAY"]:
        fmt, val, occ = recognizer.parse_and_validate_plate(sample)
        print(f"Input: '{sample:<10}' -> Hasil: '{fmt:<10}' | Valid={val} | Terhalang={occ}")
    input("\nTekan Enter untuk menutup jendela ini...")
