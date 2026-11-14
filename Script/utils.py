import os
import cv2
import numpy as np


def draw_plate_annotations(image, detections):
    """
    Menggambar anotasi visual lengkap pada citra kendaraan:
    1. Bounding box di sekeliling plat nomor (Hijau=Utuh & Pajak Aktif, Merah=Pajak Mati, Oranye=Terhalang).
    2. Garis penunjuk siku (Leader line) dari kotak plat menuju badge informasi.
    3. Badge teks 2 baris:
       - Baris 1: Nomor Plat Utama (atau peringatan TERHALANG)
       - Baris 2: Warna Pelat (Jenis Kendaraan) | Masa Pajak & Status (AKTIF / MATI)
    """
    annotated = image.copy()
    img_h, img_w = annotated.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        text = det.get("text", "PLAT")
        is_complete = det.get("is_complete", True)
        is_occluded = det.get("is_occluded", False)

        plate_color = det.get("plate_color", "-")
        tax_text = det.get("tax_text", "TIDAK TERBACA")
        tax_status = det.get("tax_status", "TIDAK TERDETEKSI")
        is_tax_active = det.get("is_tax_active", None)

        # Tentukan warna indikator utama:
        # - Oranye (0, 140, 255) jika terhalang / tidak utuh
        # - Merah terang (40, 50, 235) jika pajak terdeteksi MATI
        # - Hijau cerah (0, 220, 50) jika utuh & valid
        if not is_complete or is_occluded:
            color = (0, 140, 255)
            line1 = f"{text} (TERHALANG)" if text != "TIDAK TERBACA" else "TERHALANG / TIDAK UTUH"
        elif is_tax_active is False:
            color = (45, 55, 235)
            line1 = text
        else:
            color = (0, 220, 50)
            line1 = text

        # Susun Baris 2: Ringkasan Warna Pelat & Pajak
        info_parts = []
        if plate_color and plate_color != "-":
            info_parts.append(f"Plat {plate_color}")
        if tax_text and tax_text != "TIDAK TERBACA":
            short_status = "AKTIF" if is_tax_active else "MATI"
            info_parts.append(f"Pajak: {tax_text} ({short_status})")
        line2 = " | ".join(info_parts) if info_parts else ""

        # 1. Gambar Bounding Box mengelilingi plat
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        # 2. Gambar Garis Penunjuk (Leader line)
        start_pt = (int((x1 + x2) / 2), y1)
        line_offset_x = 70 if (start_pt[0] + 150 < img_w) else -70
        line_offset_y = -55 if (start_pt[1] - 75 > 30) else 55

        elbow_pt = (start_pt[0] + line_offset_x, start_pt[1] + line_offset_y)
        end_pt = (elbow_pt[0] + (60 if line_offset_x > 0 else -60), elbow_pt[1])

        cv2.line(annotated, start_pt, elbow_pt, color, 2, cv2.LINE_AA)
        cv2.line(annotated, elbow_pt, end_pt, color, 2, cv2.LINE_AA)
        cv2.circle(annotated, start_pt, 4, color, -1, cv2.LINE_AA)

        # 3. Gambar Kotak Badge Label Teks di ujung garis
        font = cv2.FONT_HERSHEY_SIMPLEX
        (w1, h1), _ = cv2.getTextSize(line1, font, 0.7, 2)
        (w2, h2), _ = cv2.getTextSize(line2, font, 0.48, 1) if line2 else ((0, 0), 0)

        box_w = max(w1, w2) + 20
        box_h = h1 + (h2 + 12 if line2 else 0) + 18

        if line_offset_x > 0:
            badge_x1 = min(img_w - box_w - 5, end_pt[0] + 5)
        else:
            badge_x1 = max(5, end_pt[0] - box_w - 5)

        badge_y1 = max(5, min(img_h - box_h - 5, end_pt[1] - box_h // 2))
        badge_x2 = badge_x1 + box_w
        badge_y2 = badge_y1 + box_h

        # Latar belakang gelap agar teks terbaca jelas
        cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), (20, 20, 20), -1)
        cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), color, 2)

        # Tulis Baris 1 (Nomor Plat)
        cv2.putText(annotated, line1, (badge_x1 + 10, badge_y1 + h1 + 8),
                    font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Tulis Baris 2 (Warna Pelat & Status Pajak) jika ada
        if line2:
            sub_color = (120, 245, 150) if is_tax_active is not False else (120, 140, 255)
            cv2.putText(annotated, line2, (badge_x1 + 10, badge_y1 + h1 + h2 + 18),
                        font, 0.48, sub_color, 1, cv2.LINE_AA)

    return annotated


def get_ground_truth(image_filename, base_project_dir):
    """
    Mencari nilai Ground Truth dari folder labelswithLP (jika gambar berasal dari dataset).
    Format isi file: class_id x_center y_center width height STRING_PLAT
    """
    stem = os.path.splitext(image_filename)[0]

    candidate_dirs = [
        os.path.join(base_project_dir, "Dataset", "Raw", "Indonesian License Plate Dataset", "labelswithLP"),
        os.path.join(base_project_dir, "Dataset", "images", "Indonesian License Plate Dataset", "labelswithLP"),
    ]

    import glob
    if os.path.exists("/content/dataset"):
        found_lp = [d for d in glob.glob("/content/dataset/**/labelswithLP", recursive=True) if "Recognition" not in d]
        candidate_dirs.extend(found_lp)

    splits = ["test", "train", "val"]

    for d_dir in candidate_dirs:
        if not os.path.exists(d_dir):
            continue
        for split in splits:
            txt_path = os.path.join(d_dir, split, f"{stem}.txt")
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 6:
                            return " ".join(parts[5:])
    return None


def save_step_output(img, output_folder, filename):
    """
    Helper untuk menyimpan file hasil proses ke folder Output/ jika user mengaktifkan opsi simpan.
    """
    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, filename)
    cv2.imwrite(save_path, img)
    return save_path


# ----------------------------------------------------------------------
# Fallback Helper (Digunakan oleh app.py hanya jika modul Zulfan/Ucup
# belum di-update, sehingga app.py tetap berjalan mulus 100% setiap saat)
# ----------------------------------------------------------------------
def fallback_color_and_polarity(crop_bgr):
    """
    Fallback kompatibilitas jika `detector.py` belum mengembalikan `plate_color` & `polarity_std`.
    """
    h, w = crop_bgr.shape[:2]
    roi = crop_bgr[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)]
    if roi.size == 0:
        roi = crop_bgr
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    med_s = float(np.median(s_ch))
    med_v = float(np.median(v_ch))
    med_h = float(np.median(h_ch))

    if med_s >= 65:
        if 14 <= med_h <= 42:
            plate_color = "Kuning"
            vehicle_type = "Kendaraan Angkutan Umum / Barang"
        elif med_h <= 13 or med_h >= 160:
            plate_color = "Merah"
            vehicle_type = "Kendaraan Dinas Pemerintah"
        else:
            plate_color = "Putih" if med_v > 127 else "Hitam"
            vehicle_type = f"Kendaraan Pribadi (Pelat {plate_color})"
    else:
        if med_v >= 125:
            plate_color = "Putih"
            vehicle_type = "Kendaraan Pribadi (Pelat Putih)"
        else:
            plate_color = "Hitam"
            vehicle_type = "Kendaraan Pribadi (Pelat Hitam)"

    is_inverted = (plate_color in ("Putih", "Kuning")) or (med_v >= 127)
    polarity_std = cv2.bitwise_not(crop_bgr) if is_inverted else crop_bgr.copy()
    pol_status = (
        f"Inversi Aktif (Plat {plate_color} -> Latar Hitam, V={med_v:.0f})"
        if is_inverted else f"Tetap / Asli (Plat {plate_color} Latar Hitam, V={med_v:.0f})"
    )
    return {
        "plate_color": plate_color,
        "vehicle_type": vehicle_type,
        "polarity_std": polarity_std,
        "is_inverted": is_inverted,
        "polarity_status": pol_status,
        "hsv_summary": f"H={med_h:.0f}, S={med_s:.0f}, V={med_v:.0f}"
    }


def fallback_extract_tax_roi(gray_img, binary_img):
    """
    Fallback kompatibilitas jika `binarizer.py` belum mengembalikan `tax_roi_binary` & `tax_roi_gray`.
    Memotong area tengah-bawah pelat (y: 62%-93%, x: 22%-78%) untuk nomor pajak MM.YY.
    """
    gh, gw = gray_img.shape[:2]
    y1, y2 = int(gh * 0.62), int(gh * 0.93)
    x1, x2 = int(gw * 0.22), int(gw * 0.78)
    tax_gray = gray_img[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else gray_img

    if tax_gray.size > 0:
        tax_gray = cv2.resize(tax_gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        _, tax_bin = cv2.threshold(tax_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Pastikan standar latar hitam (0) & teks putih (255)
        if np.mean(tax_bin == 255) > 0.5:
            tax_bin = cv2.bitwise_not(tax_bin)
    else:
        tax_bin = binary_img
    return tax_bin, tax_gray
