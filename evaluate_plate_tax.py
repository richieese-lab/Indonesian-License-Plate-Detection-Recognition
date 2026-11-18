import os
import re
import glob
import argparse
import cv2
import easyocr

from Script.plate_tax_pipeline import (
    find_recognition_dataset,
    load_class_names,
    analyze_single_plate,
    load_gt_from_txt,
    evaluate_ocr_metrics,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluasi Pipeline Nomor Plat & Angka Pajak (Tanpa YOLO)")
    parser.add_argument("--dataset", type=str, default=None, help="Path ke Indonesian License Plate Recognition Dataset")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "val"], help="Split dataset")
    parser.add_argument("--max-samples", type=int, default=30, help="Jumlah maksimal sampel (0 = semua)")
    parser.add_argument("--files", nargs="*", default=None, help="Daftar nama file spesifik (misal train002_2.jpg)")
    parser.add_argument("--no-gpu", action="store_true", help="Paksa EasyOCR menggunakan CPU")
    parser.add_argument("--use-vlm", action="store_true", help="Aktifkan verifikasi konsensus dengan AI lokal (qwen3.5:9b)")
    args = parser.parse_args()

    dataset_root = find_recognition_dataset(args.dataset)
    if not dataset_root:
        raise FileNotFoundError(
            "Dataset tidak ditemukan! Jalankan download_dataset atau pastikan kagglehub cache tersedia."
        )

    class_names = load_class_names(dataset_root)
    reader = easyocr.Reader(["en"], gpu=not args.no_gpu)

    if args.files:
        eval_imgs = []
        for fname in args.files:
            matches = glob.glob(os.path.join(dataset_root, "images", "**", fname), recursive=True)
            if matches:
                eval_imgs.append(matches[0])
            else:
                print(f"[PERINGATAN] File tidak ditemukan: {fname}")
    else:
        img_dir = os.path.join(dataset_root, "images", args.split)
        eval_imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        if args.max_samples and args.max_samples > 0:
            eval_imgs = eval_imgs[: args.max_samples]

    if not eval_imgs:
        print("Tidak ada citra yang dapat dievaluasi.")
        return

    valid_preds, valid_gts = [], []
    invalid_count = 0
    tax_valid_count = 0

    print(f"Dataset: {dataset_root}")
    print(f"Jumlah sampel: {len(eval_imgs)}\n")

    for idx, path in enumerate(eval_imgs, 1):
        bgr = cv2.imread(path)
        res = analyze_single_plate(bgr, reader, use_vlm=args.use_vlm)
        gt_txt = load_gt_from_txt(path, class_names)
        fname = os.path.basename(path)

        if not res["plate_valid"]:
            invalid_count += 1
            print(
                f"[{idx:02d}/{len(eval_imgs):02d}] {fname:<16} | {res['plate_status']:<40} | "
                f"GT: {gt_txt:<11} | Pajak: {res['tax_text']} ({res['tax_status']})"
            )
            continue

        gt_c = re.sub(r"[^A-Z0-9]", "", gt_txt)
        pr_c = re.sub(r"[^A-Z0-9]", "", res["plate_text"])
        sym = "OK" if gt_c == pr_c else "MISS"
        if res["tax_valid"]:
            tax_valid_count += 1

        print(
            f"[{idx:02d}/{len(eval_imgs):02d}] {fname:<16} | Pred: {res['plate_text']:<11} ({res['plate_conf']:.2f}) | "
            f"GT: {gt_txt:<11} | Pajak: {res['tax_text']:<6} ({res['tax_conf']:.2f}) | {sym}"
        )
        valid_preds.append(res["plate_text"])
        valid_gts.append(gt_txt)

    print("\n" + "=" * 72)
    print(f"RINGKASAN EVALUASI | Valid: {len(valid_preds)} | Tidak Valid / Ditolak: {invalid_count}")
    print("=" * 72)
    if valid_preds:
        metrics = evaluate_ocr_metrics(valid_preds, valid_gts)
        for k, v in metrics.items():
            print(f"  {k:<28}: {v:.2f}%")
        print(f"  {'Pajak Valid Terbaca (%)':<28}: {(tax_valid_count / len(valid_preds)) * 100:.2f}%")
    else:
        print("  Tidak ada sampel valid untuk dihitung metriknya.")
    print("=" * 72)


if __name__ == "__main__":
    main()
