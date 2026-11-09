import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Import modul lokal dari folder Script/
from Script.detector import PlateDetector
from Script.preprocessor import PlatePreprocessor
from Script.binarizer import PlateBinarizer
from Script.recognizer import PlateRecognizer
from Script.utils import draw_plate_annotations, get_ground_truth, save_step_output

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Sistem Deteksi & Pengenalan Pelat Nomor",
    page_icon="🚗",
    layout="wide"
)

# Inisialisasi Path Dasar Project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "Output")

# Deteksi lokasi folder images/test (mendukung Lokal maupun Google Colab)
TEST_SET_DIR = os.path.join(BASE_DIR, "Dataset", "Raw", "Indonesian License Plate Dataset", "images", "test")
if not os.path.exists(TEST_SET_DIR):
    import glob
    candidates = [d for d in glob.glob("/content/dataset/**/images/test", recursive=True) if "Recognition" not in d]
    if candidates:
        TEST_SET_DIR = candidates[0]

# Cache Model agar tidak reload setiap kali user berinteraksi
@st.cache_resource
def load_models():
    detector = PlateDetector()
    recognizer = PlateRecognizer(use_gpu=False)
    preprocessor = PlatePreprocessor()
    return detector, recognizer, preprocessor

# Header & Info Proyek
st.title("🚗 Sistem Deteksi & Pengenalan Pelat Nomor Kendaraan Indonesia")
st.markdown("""
Sistem ini mengintegrasikan materi kuliah **Pengolahan Citra Digital dan Visi Komputer**:
* **P21. YOLO:** Deteksi dan Lokalisasi Kotak Plat Nomor pada Kendaraan.
* **P4. Ruang Warna & P6. Histogram CLAHE:** Peningkatan Kontras Adaptif Plat.
* **P12. Segmentasi Dasar:** Binarisasi Karakter (Adaptive / Otsu Thresholding).
* **EasyOCR & Filter Oklusi:** Pembacaan Teks dan Deteksi Plat Terhalang / Tidak Utuh.
""")
st.divider()

# Sidebar: Pengaturan dan Input
st.sidebar.header("⚙️ Kontrol & Pengaturan")

input_source = st.sidebar.radio(
    "Pilih Sumber Gambar:",
    ("Unggah Gambar Sendiri (Termasuk Foto Terhalang)", "Pilih dari Dataset Uji (images/test)")
)

uploaded_file = None
selected_sample_name = None

if input_source == "Unggah Gambar Sendiri (Termasuk Foto Terhalang)":
    uploaded_file = st.sidebar.file_uploader(
        "Upload foto kendaraan (JPG/PNG)", 
        type=["jpg", "jpeg", "png"]
    )
else:
    sample_files = [f for f in os.listdir(TEST_SET_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))] if os.path.exists(TEST_SET_DIR) else []
    if sample_files:
        selected_sample_name = st.sidebar.selectbox("Pilih file sampel uji:", sample_files)
    else:
        st.sidebar.warning("Folder images/test tidak ditemukan.")

st.sidebar.subheader("Parameter Pemrosesan")
conf_thresh = st.sidebar.slider("Confidence Threshold YOLO:", 0.1, 0.9, 0.35, 0.05)
binarize_method = st.sidebar.selectbox("Metode Binarisasi (P12):", ["otsu", "adaptive"])

show_pcd_steps = st.sidebar.checkbox("Tampilkan Langkah Pengolahan Citra (PCD)", value=True)
save_to_output = st.sidebar.checkbox("Simpan Hasil ke Folder Output/", value=False)

# Muat Gambar
image_bgr = None
image_name = "sample.jpg"

if input_source == "Unggah Gambar Sendiri (Termasuk Foto Terhalang)" and uploaded_file is not None:
    image_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    image_name = uploaded_file.name
elif input_source == "Pilih dari Dataset Uji (images/test)" and selected_sample_name:
    sample_path = os.path.join(TEST_SET_DIR, selected_sample_name)
    image_bgr = cv2.imread(sample_path)
    image_name = selected_sample_name

if image_bgr is not None:
    # Memuat modul
    detector, recognizer, preprocessor = load_models()
    binarizer = PlateBinarizer(method=binarize_method)
    detector.conf_threshold = conf_thresh

    # Tombol Eksekusi
    if st.button("🚀 Proses Deteksi & Pembacaan Pelat", type="primary"):
        with st.spinner("Sedang memproses citra melalui pipeline..."):
            # 1. Tahap Deteksi (P21 YOLO)
            detections = detector.detect(image_bgr)

            if not detections:
                st.warning("⚠️ Tidak ditemukan plat nomor pada gambar ini. Coba turunkan nilai Confidence Threshold di sidebar.")
                st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Gambar Asli", use_container_width=True)
            else:
                pcd_results = []
                
                # 2. Loop setiap plat nomor yang terdeteksi
                for i, det in enumerate(detections):
                    crop_img = det["crop"]
                    
                    # Simpan crop jika diaktifkan
                    if save_to_output:
                        save_step_output(crop_img, os.path.join(OUTPUT_DIR, "01_cropped_plates"), f"crop_{i+1}_{image_name}")

                    # 3. Preprocessing (P4 Grayscale + P6 CLAHE)
                    prep_res = preprocessor.process(crop_img)
                    enhanced_gray = prep_res["enhanced"]
                    
                    if save_to_output:
                        save_step_output(enhanced_gray, os.path.join(OUTPUT_DIR, "02_preprocessed"), f"clahe_{i+1}_{image_name}")

                    # 4. Binarisasi (P12 Thresholding)
                    bin_res = binarizer.process(enhanced_gray)
                    binary_norm = bin_res["binary_normalized"]
                    
                    if save_to_output:
                        save_step_output(binary_norm, os.path.join(OUTPUT_DIR, "03_binarized"), f"binary_{i+1}_{image_name}")

                    # 5. Pembacaan Teks (EasyOCR dengan multi-pass fallback)
                    ocr_res = recognizer.recognize(binary_norm, fallback_image=bin_res["upscaled_gray"], extra_images=bin_res["variants"][:1])

                    plate_text = ocr_res["clean_text"]
                    det["text"] = plate_text
                    det["is_occluded"] = ocr_res["is_occluded"] or (not det["is_complete"]) or (not ocr_res["is_valid_format"]) or ocr_res["confidence"] < 0.4

                    # Cek Ground Truth
                    gt_text = get_ground_truth(image_name, BASE_DIR)

                    pcd_results.append({
                        "index": i + 1,
                        "crop": crop_img,
                        "grayscale": prep_res["grayscale"],
                        "clahe": enhanced_gray,
                        "binary": binary_norm,
                        "text": plate_text,
                        "confidence": ocr_res["confidence"],
                        "aspect_ratio": det["aspect_ratio"],
                        "is_complete": det["is_complete"],
                        "status": det["status"],
                        "is_occluded": det["is_occluded"],
                        "ground_truth": gt_text,
                        "clahe_status": prep_res.get("clahe_status", "Aktif")
                    })

                # 6. Gambar Anotasi Visual (Kotak, Garis Penunjuk, Badge Teks)
                annotated_img = draw_plate_annotations(image_bgr, detections)

                if save_to_output:
                    save_step_output(annotated_img, os.path.join(OUTPUT_DIR, "04_final_detections"), f"final_{image_name}")

                # Tampilkan Hasil Utama
                col_main, col_info = st.columns([2, 1])

                with col_main:
                    st.subheader("🖼️ Hasil Deteksi & Visualisasi Garis Penunjuk")
                    st.image(
                        cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), 
                        caption="Gambar Beranotasi (Bounding Box + Garis Penunjuk Plat)", 
                        use_container_width=True
                    )

                with col_info:
                    st.subheader("📋 Informasi & Hasil Bacaan")
                    for res in pcd_results:
                        st.markdown(f"### Plat #{res['index']}: **{res['text']}**")
                        
                        if res["is_occluded"] or not res["is_complete"]:
                            st.error(f"⚠️ **Status: Terhalang / Tidak Utuh!** ({res['status']})")
                        else:
                            st.success("✅ **Status: Utuh & Valid**")

                        st.write(f"• **Confidence OCR:** {res['confidence']}")
                        st.write(f"• **Rasio Aspek (W/H):** {res['aspect_ratio']}")
                        st.write(f"• **Analisis Histogram (P6):** {res['clahe_status']}")

                        if res["ground_truth"]:
                            st.info(f"🎯 **Ground Truth:** `{res['ground_truth']}`")
                            is_match = (res['text'].replace(" ", "") == res['ground_truth'].replace(" ", ""))
                            if is_match:
                                st.success("🎉 Cocok 100% dengan Kunci Jawaban!")
                            else:
                                st.warning("🔍 Terdapat perbedaan karakter dengan Ground Truth.")
                        
                        st.divider()

                # 7. Langkah PCD Interaktif (Jika Checkbox Dicentang)
                if show_pcd_steps:
                    st.subheader("🔬 Visualisasi Langkah Pengolahan Citra Digital (PCD)")
                    
                    for res in pcd_results:
                        st.markdown(f"**Detail Tahapan Plat #{res['index']}:**")
                        t1, t2, t3, t4 = st.columns(4)

                        with t1:
                            st.caption("1. Crop Plat Asli (P21 YOLO)")
                            st.image(cv2.cvtColor(res["crop"], cv2.COLOR_BGR2RGB), use_container_width=True)

                        with t2:
                            st.caption("2. Grayscale (Materi P4)")
                            st.image(res["grayscale"], clamp=True, use_container_width=True)

                        with t3:
                            st.caption("3. Histogram CLAHE (Materi P6)")
                            st.image(res["clahe"], clamp=True, use_container_width=True)

                        with t4:
                            st.caption("4. Segmentasi Biner (Materi P12)")
                            st.image(res["binary"], clamp=True, use_container_width=True)

                if save_to_output:
                    st.success(f"📁 Seluruh hasil proses telah disimpan ke folder `{OUTPUT_DIR}`!")

else:
    st.info("👈 Silakan unggah gambar kendaraan atau pilih salah satu sampel uji coba dari sidebar untuk memulai.")
