import os
import cv2
import numpy as np
from ultralytics import YOLO

class PlateDetector:
    """
    Modul Deteksi Plat Nomor Kendaraan (Materi P21: YOLO).
    Mendeteksi posisi bounding box plat nomor dan melakukan cropping ROI.
    Dilengkapi deteksi rasio aspek untuk mendeteksi plat yang terhalang/tidak utuh.
    """
    def __init__(self, model_path=None, conf_threshold=0.35):
        self.conf_threshold = conf_threshold
        
        # Cari bobot model: prioritas models/best.pt -> models/license_plate_detector.pt -> yolov8n.pt
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_model = os.path.join(base_dir, "models", "best.pt")
        alt_model = os.path.join(base_dir, "models", "license_plate_detector.pt")
        
        if model_path and os.path.exists(model_path):
            self.model_path = model_path
        elif os.path.exists(default_model):
            self.model_path = default_model
        elif os.path.exists(alt_model):
            self.model_path = alt_model
        else:
            # Fallback ke pretrained YOLOv8n (akan di-download otomatis oleh Ultralytics jika belum ada)
            self.model_path = "yolov8n.pt"
            
        print(f"[PlateDetector] Memuat model YOLO dari: {self.model_path}")
        self.model = YOLO(self.model_path)

    def detect(self, image_input):
        """
        Mendeteksi plat nomor dari gambar.
        Input: image_input (bisa berupa path file str atau numpy.ndarray BGR)
        Output: list of dict berisi koordinat box, confidence, citra crop, dan status kelengkapan.
        """
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Gagal membaca gambar dari: {image_input}")
        else:
            image = image_input.copy()

        img_h, img_w = image.shape[:2]
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())

                x1, y1, x2, y2 = coords
                # Pastikan koordinat berada di dalam batas ukuran citra
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_w, x2)
                y2 = min(img_h, y2)

                w = x2 - x1
                h = y2 - y1

                if w <= 10 or h <= 10:
                    continue

                aspect_ratio = float(w) / float(h)
                
                # Standar plat nomor Indonesia memiliki rasio panjang:tinggi sekitar 2.0 - 4.5.
                # Jika rasio terlalu kecil (< 1.8) atau terlalu tinggi (> 5.5), plat kemungkinan
                # terpotong oklusi (terhalang baju/helm/kendaraan lain) atau deteksi salah.
                is_complete = True
                status_desc = "Lengkap"

                if aspect_ratio < 1.8:
                    is_complete = False
                    status_desc = "Terhalang / Tidak Utuh (Rasio Terlalu Kecil)"
                elif aspect_ratio > 5.5:
                    is_complete = False
                    status_desc = "Tidak Wajar (Rasio Terlalu Panjang)"

                # Potong citra plat nomor (Crop ROI)
                crop_img = image[y1:y2, x1:x2]

                detections.append({
                    "box": (x1, y1, x2, y2),
                    "confidence": conf,
                    "class_id": cls_id,
                    "crop": crop_img,
                    "aspect_ratio": round(aspect_ratio, 2),
                    "is_complete": is_complete,
                    "status": status_desc
                })

        return detections

if __name__ == "__main__":
    # Test Mandiri Script detector.py
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_img_path = os.path.join(base_dir, "Dataset", "Raw", "Indonesian License Plate Dataset", "images", "test", "test001.jpg")
    output_crop_dir = os.path.join(base_dir, "Output", "01_cropped_plates")
    os.makedirs(output_crop_dir, exist_ok=True)

    print(f"Menjalankan tes mandiri pada: {test_img_path}")
    detector = PlateDetector()
    results = detector.detect(test_img_path)

    print(f"Total objek terdeteksi: {len(results)}")
    for i, det in enumerate(results):
        print(f"Deteksi #{i+1}: Box={det['box']}, Conf={det['confidence']:.2f}, "
              f"Rasio Aspek={det['aspect_ratio']}, Status={det['status']}")
        crop_save_path = os.path.join(output_crop_dir, f"crop_test_{i+1}.jpg")
        cv2.imwrite(crop_save_path, det["crop"])
        print(f"   -> Hasil crop disimpan ke: {crop_save_path}")
    input("\nTekan Enter untuk menutup jendela ini...")
