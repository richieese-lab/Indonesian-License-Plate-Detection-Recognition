import os
import sys
import glob
import cv2
import easyocr

from Script.plate_tax_pipeline import (
    find_recognition_dataset,
    analyze_single_plate,
)

# ==============================================================================
# GANTI PATH FOTO DI BAWAH INI UNTUK MENGUJI SATU GAMBAR
# ==============================================================================
IMAGE_PATH = ""  # Contoh: "/path/ke/foto_plat.jpg" atau "Dataset/sample.jpg"
USE_LOCAL_VLM = True  # True = verifikasi konsensus dengan qwen3.5:9b (Ollama)
# ==============================================================================


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else IMAGE_PATH

    if not img_path:
        dataset_root = find_recognition_dataset()
        if dataset_root:
            img_path = os.path.join(dataset_root, "images", "test", "test003_1.jpg")
        else:
            print("Silakan isi variabel IMAGE_PATH di dalam file atau berikan path via argumen.")
            return

    if not os.path.exists(img_path):
        # Coba cari otomatis berdasarkan nama file di dataset jika path lengkap tidak ditemukan
        fname = os.path.basename(img_path)
        dataset_root = find_recognition_dataset()
        found = False
        if dataset_root:
            matches = glob.glob(os.path.join(dataset_root, "images", "**", fname), recursive=True)
            if matches:
                img_path = matches[0]
                found = True
        if not found:
            print(f"File tidak ditemukan: {img_path}")
            return

    bgr = cv2.imread(img_path)
    if bgr is None:
        print(f"Gagal membaca gambar: {img_path}")
        return

    reader = easyocr.Reader(["en"], gpu=True)
    res = analyze_single_plate(bgr, reader, use_vlm=USE_LOCAL_VLM)

    print("\n" + "=" * 55)
    print(f"File        : {os.path.basename(img_path)}")
    print(f"Path        : {img_path}")
    print("-" * 55)
    print(f"Nomor Plat  : {res['plate_text']}")
    print(f"Status Plat : {res['plate_status']} (conf={res['plate_conf']:.2f})")
    print(f"Angka Pajak : {res['tax_text']}")
    print(f"Status Pajak: {res['tax_status']} (conf={res['tax_conf']:.2f})")
    print(f"Ketajaman   : {res['sharpness_lap']}")
    print("=" * 55)


if __name__ == "__main__":
    main()
