import os
import cv2
import numpy as np

def draw_plate_annotations(image, detections):
    """
    Menggambar anotasi visual sesuai permintaan:
    1. Bounding box di sekeliling plat nomor.
    2. Garis penunjuk dari sudut kotak plat menuju label luar.
    3. Badge teks di ujung garis berisi angka dan huruf plat (atau status jika terhalang).
    """
    annotated = image.copy()
    img_h, img_w = annotated.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        text = det.get("text", "PLAT")
        is_complete = det.get("is_complete", True)
        is_occluded = det.get("is_occluded", False)

        # Tentukan warna:
        # Hijau (0, 220, 0) jika lengkap & valid
        # Oranye / Kuning (0, 165, 255) jika terhalang / tidak utuh
        if not is_complete or is_occluded:
            color = (0, 140, 255) # Oranye/Kuning peringatan
            display_text = f"{text} (TERHALANG)" if text != "TIDAK TERBACA" else "TERHALANG / TIDAK UTUH"
        else:
            color = (0, 220, 50)  # Hijau cerah
            display_text = text

        # 1. Gambar Bounding Box mengelilingi plat
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        # 2. Gambar Garis Penunjuk (Leader line)
        # Ambil titik tengah sisi atas kotak plat
        start_pt = (int((x1 + x2) / 2), y1)
        
        # Tentukan titik ujung garis (ke atas-kanan jika cukup ruang, atau ke atas-kiri)
        line_offset_x = 70 if (start_pt[0] + 120 < img_w) else -70
        line_offset_y = -50 if (start_pt[1] - 60 > 30) else 50
        
        elbow_pt = (start_pt[0] + line_offset_x, start_pt[1] + line_offset_y)
        end_pt = (elbow_pt[0] + (60 if line_offset_x > 0 else -60), elbow_pt[1])

        # Gambar garis bercabang (elbow line)
        cv2.line(annotated, start_pt, elbow_pt, color, 2, cv2.LINE_AA)
        cv2.line(annotated, elbow_pt, end_pt, color, 2, cv2.LINE_AA)
        cv2.circle(annotated, start_pt, 4, color, -1, cv2.LINE_AA)

        # 3. Gambar Kotak Badge Label Teks di ujung garis
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (txt_w, txt_h), baseline = cv2.getTextSize(display_text, font, font_scale, thickness)

        if line_offset_x > 0:
            badge_x1 = end_pt[0] + 5
            badge_y1 = end_pt[1] - txt_h - 10
        else:
            badge_x1 = end_pt[0] - txt_w - 20
            badge_y1 = end_pt[1] - txt_h - 10

        badge_x2 = badge_x1 + txt_w + 16
        badge_y2 = badge_y1 + txt_h + 16

        # Latar belakang gelap agar teks sangat terbaca jelas
        cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), (20, 20, 20), -1)
        # Bingkai badge dengan warna indikator
        cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), color, 2)
        # Tulis teks plat
        cv2.putText(annotated, display_text, (badge_x1 + 8, badge_y2 - 8), 
                    font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return annotated

def get_ground_truth(image_filename, base_project_dir):
    """
    Mencari nilai Ground Truth dari folder labelswithLP (jika gambar berasal dari dataset).
    Format isi file: class_id x_center y_center width height STRING_PLAT
    Contoh: 0 0.59 0.71 0.10 0.03 G9058EJ -> mengembalikan 'G9058EJ'
    """
    stem = os.path.splitext(image_filename)[0]
    
    # Lokasi folder labelswithLP di Dataset (mendukung Lokal dan Google Colab)
    dataset_dir = os.path.join(base_project_dir, "Dataset", "Raw", "Indonesian License Plate Dataset", "labelswithLP")
    candidate_dirs = [dataset_dir]
    
    if not os.path.exists(dataset_dir):
        import glob
        found_lp = [d for d in glob.glob("/content/dataset/**/labelswithLP", recursive=True) if "Recognition" not in d]
        if found_lp:
            candidate_dirs.append(found_lp[0])
    
    splits = ["test", "train", "val"]
    
    for d_dir in candidate_dirs:
        if not os.path.exists(d_dir):
            continue
        for split in splits:
            txt_path = os.path.join(d_dir, split, f"{stem}.txt")
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in lines:
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
