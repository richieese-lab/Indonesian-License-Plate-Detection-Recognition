"""
Bandingkan akurasi OCR plat_tax_pipeline versi lama (branch main) vs versi baru (file di Script/).

Alurnya sama seperti di app.py: crop plat -> detector.py (deskew, warna) -> analyze_single_plate.
Kotak plat diambil dari label dataset lalu dilonggarkan secara acak supaya mirip kotak YOLO.

Pemakaian (dari folder proyek, venv aktif):
    python uji_ocr_banding.py            # 40 plat pertama data test
    python uji_ocr_banding.py --n 100    # lebih banyak plat, lebih lama
"""
import os
import re
import sys
import glob
import types
import argparse
import subprocess

import cv2
import numpy as np
import easyocr

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from Script.detector import PlateDetector
from Script import plate_tax_pipeline as baru


def load_versi_lama(ref="main"):
    """Ambil plate_tax_pipeline.py dari branch main tanpa mengubah file di folder."""
    src = subprocess.run(["git", "show", f"{ref}:Script/plate_tax_pipeline.py"],
                         cwd=BASE, capture_output=True, text=True, encoding="utf-8")
    if src.returncode != 0:
        sys.exit(f"Gagal membaca versi lama dari git ({ref}): {src.stderr.strip()}")
    mod = types.ModuleType("plate_tax_pipeline_lama")
    exec(compile(src.stdout, "plate_tax_pipeline_lama.py", "exec"), mod.__dict__)
    return mod


def ambil_crop(n, seed):
    root = os.path.join(BASE, "Dataset", "Raw", "Indonesian License Plate Dataset")
    rng = np.random.default_rng(seed)
    items = []
    for ip in sorted(glob.glob(os.path.join(root, "images", "test", "*.jpg"))):
        stem = os.path.splitext(os.path.basename(ip))[0]
        lab = os.path.join(root, "labelswithLP", "test", stem + ".txt")
        if not os.path.exists(lab):
            continue
        img = cv2.imread(ip)
        H, W = img.shape[:2]
        for k, line in enumerate(open(lab, encoding="utf-8")):
            p = line.split()
            if len(p) < 6:
                continue
            cx, cy, bw, bh = map(float, p[1:5])
            t, b, s = rng.uniform(.10, .40), rng.uniform(.05, .25), rng.uniform(.04, .12)
            x1, x2 = int((cx - bw / 2 - bw * s) * W), int((cx + bw / 2 + bw * s) * W)
            y1, y2 = int((cy - bh / 2 - bh * t) * H), int((cy + bh / 2 + bh * b) * H)
            crop = img[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
            if crop.size:
                items.append((f"{stem}_{k}", "".join(p[5:]).upper(), crop))
            if len(items) >= n:
                return items
    return items


def norm(s):
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    lama = load_versi_lama()
    det = PlateDetector()
    try:
        import torch
        gpu = torch.cuda.is_available()
    except Exception:
        gpu = False
    reader = easyocr.Reader(["en"], gpu=gpu)

    items = ambil_crop(args.n, args.seed)
    print(f"\nJumlah plat uji: {len(items)} (OCR {'GPU' if gpu else 'CPU'}, bisa beberapa menit)\n")

    hasil = {"lama": [], "baru": []}
    for i, (name, gt, crop) in enumerate(items, 1):
        deskewed = det.process_crop(crop)["deskewed"]
        r_lama = lama.analyze_single_plate(deskewed, reader)                      # perilaku app lama
        r_baru = baru.analyze_single_plate(deskewed, reader, skip_deskew=True)    # perilaku app baru
        hasil["lama"].append((gt, r_lama))
        hasil["baru"].append((gt, r_baru))
        tanda = ""
        if norm(r_lama["plate_text"]) != norm(r_baru["plate_text"]) or r_lama["tax_text"] != r_baru["tax_text"]:
            tanda = "  <- beda"
        print(f"[{i:3d}/{len(items)}] {name:12s} GT {gt:10s} | lama: {r_lama['plate_text']:13s} pajak {r_lama['tax_text']:6s}"
              f" | baru: {r_baru['plate_text']:13s} pajak {r_baru['tax_text']:6s}{tanda}")

    print("\n================ RINGKASAN ================")
    for versi, rows in hasil.items():
        n = len(rows)
        exact = sum(norm(r["plate_text"]) == norm(gt) for gt, r in rows)
        cer_e = sum(baru.levenshtein_distance(norm(r["plate_text"]), norm(gt)) for gt, r in rows)
        cer_n = sum(max(1, len(norm(gt))) for gt, _ in rows)
        pajak = sum(r["tax_valid"] for _, r in rows)
        print(f"{versi:5s}: nomor plat tepat {exact}/{n} ({exact / n:.0%}) | CER {cer_e / cer_n:.1%} | "
              f"angka pajak terbaca format MM-YY {pajak}/{n} ({pajak / n:.0%})")
    print("\nCatatan: dataset tidak punya label angka pajak, jadi yang dihitung hanya berapa yang berhasil"
          " terbaca dengan format valid. Cek kebenarannya dengan melihat foto platnya.")


if __name__ == "__main__":
    main()
