import os
import cv2


def draw_plate_annotations(image, detections):
    """
    Menggambar bounding box, garis penunjuk, dan badge informasi pelat + status pajak.
    Warna indikator:
    - Hijau (0, 220, 80)  : Pelat utuh & terbaca valid
    - Oranye (0, 140, 255): Pelat terhalang / terpotong tepi gambar
    - Biru (220, 140, 40) : Pelat utuh tetapi teks belum terbaca jelas
    """
    annotated = image.copy()
    img_h, img_w = annotated.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        text = det.get("plate_text", det.get("text", "PLAT"))
        tax_text = det.get("tax_text", "-")
        is_complete = det.get("is_complete", True)
        is_occluded = det.get("is_occluded", False)
        is_valid = det.get("plate_valid", True)

        if not is_complete or is_occluded:
            color = (0, 140, 255)
            display_text = f"{text} (TERHALANG)" if text not in ("TIDAK TERBACA", "TIDAK VALID") else "TERHALANG"
        elif not is_valid:
            color = (220, 140, 40)
            display_text = text
        else:
            color = (0, 220, 80)
            display_text = text if tax_text in ("-", "TIDAK TERBACA") else f"{text} | {tax_text}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        start_pt = (int((x1 + x2) / 2), y1)
        line_offset_x = 65 if (start_pt[0] + 130 < img_w) else -65
        line_offset_y = -45 if (start_pt[1] - 55 > 25) else 45
        elbow_pt = (start_pt[0] + line_offset_x, start_pt[1] + line_offset_y)
        end_pt = (elbow_pt[0] + (50 if line_offset_x > 0 else -50), elbow_pt[1])

        cv2.line(annotated, start_pt, elbow_pt, color, 2, cv2.LINE_AA)
        cv2.line(annotated, elbow_pt, end_pt, color, 2, cv2.LINE_AA)
        cv2.circle(annotated, start_pt, 4, color, -1, cv2.LINE_AA)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (txt_w, txt_h), _ = cv2.getTextSize(display_text, font, font_scale, thickness)

        badge_x1 = end_pt[0] + 5 if line_offset_x > 0 else end_pt[0] - txt_w - 20
        badge_y1 = end_pt[1] - txt_h - 10
        badge_x1 = max(6, min(badge_x1, img_w - txt_w - 22))
        badge_y1 = max(6, min(badge_y1, img_h - txt_h - 22))
        badge_x2 = badge_x1 + txt_w + 16
        badge_y2 = badge_y1 + txt_h + 16

        cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), (17, 19, 24), -1)
        cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), color, 2)
        cv2.putText(
            annotated, display_text, (badge_x1 + 8, badge_y2 - 8),
            font, font_scale, (243, 244, 246), thickness, cv2.LINE_AA,
        )

    return annotated


def get_ground_truth(image_filename):
    """Mencari label ground truth jika file berasal dari cache Kaggle dataset."""
    stem = os.path.splitext(image_filename)[0]
    kaggle_cache_dir = os.path.expanduser(
        "~/.cache/kagglehub/datasets/juanthomaswijaya/indonesian-license-plate-dataset/versions/1/Indonesian License Plate Dataset/labelswithLP"
    )
    if not os.path.exists(kaggle_cache_dir):
        return None
    for split in ("test", "train", "val"):
        txt_path = os.path.join(kaggle_cache_dir, split, f"{stem}.txt")
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        return " ".join(parts[5:])
    return None
