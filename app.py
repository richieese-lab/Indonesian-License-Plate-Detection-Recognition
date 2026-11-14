import os
import glob
import cv2
import numpy as np
import streamlit as st

# Import modul lokal dari folder Script/
from Script.detector import PlateDetector
from Script.preprocessor import PlatePreprocessor
from Script.binarizer import PlateBinarizer
from Script.recognizer import PlateRecognizer
from Script.utils import (
    draw_plate_annotations,
    get_ground_truth,
    save_step_output,
    fallback_color_and_polarity,
    fallback_extract_tax_roi,
)

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Sistem Deteksi & Pengenalan Pelat Nomor Indonesia",
    page_icon="🚗",
    layout="wide"
)

# Inisialisasi Path Dasar Project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "Output")

# Deteksi lokasi folder images/test (mendukung Lokal di Dataset/images maupun Dataset/Raw serta Google Colab)
CANDIDATE_TEST_DIRS = [
    os.path.join(BASE_DIR, "Dataset", "images", "Indonesian License Plate Dataset", "images", "test"),
    os.path.join(BASE_DIR, "Dataset", "Raw", "Indonesian License Plate Dataset", "images", "test"),
]
TEST_SET_DIR = next((d for d in CANDIDATE_TEST_DIRS if os.path.exists(d)), CANDIDATE_TEST_DIRS[0])
if not os.path.exists(TEST_SET_DIR):
    candidates = [d for d in glob.glob("/content/dataset/**/images/test", recursive=True) if "Recognition" not in d]
    if candidates:
        TEST_SET_DIR = candidates[0]


@st.cache_resource
def load_models():
    detector = PlateDetector()
    recognizer = PlateRecognizer(use_gpu=False)
    preprocessor = PlatePreprocessor()
    return detector, recognizer, preprocessor


# Header & Info Proyek
st.title("🚗 Sistem Deteksi Pelat Nomor, Klasifikasi Kendaraan & Status Pajak (ALPR)")
st.markdown("""
Sistem ini mengintegrasikan **Alur Sistem & Box Evaluasi Pengolahan Citra Digital (PCD)** beserta **2 Kebaruan Utama**:
* **Blok YOLOv8n:** Deteksi & *Crop* ROI $\\rightarrow$ Koreksi Kemiringan (*Deskewing*) $\\rightarrow$ **Kebaruan 1: Identifikasi Warna Pelat (HSV) & Klasifikasi Jenis Kendaraan** + Standarisasi Polaritas (Background Hitam & Tulisan Putih).
* **Blok Pengenalan Teks (OCR):** *Grayscale & Bilateral Smoothing* $\\rightarrow$ *Conditional CLAHE* $\\rightarrow$ *Unsharp Masking* $\\rightarrow$ *Segmentasi Biner* $\\rightarrow$ *Perbaikan Morfologi* $\\rightarrow$ **Kebaruan 2: Pemisahan Dual-ROI (Nomor Utama & Masa Pajak `MM.YY`)** $\\rightarrow$ Multi-Input OCR, Validasi Regex & Klasifikasi **Pajak Aktif vs Mati**.
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
        "Upload foto kendaraan / pelat nomor (JPG/PNG)",
        type=["jpg", "jpeg", "png"]
    )
else:
    sample_files = sorted(
        [f for f in os.listdir(TEST_SET_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    ) if os.path.exists(TEST_SET_DIR) else []
    if sample_files:
        selected_sample_name = st.sidebar.selectbox("Pilih file sampel uji:", sample_files)
    else:
        st.sidebar.warning("Folder images/test tidak ditemukan.")

st.sidebar.subheader("Parameter Pemrosesan")
conf_thresh = st.sidebar.slider("Confidence Threshold YOLO:", 0.1, 0.9, 0.35, 0.05)
binarize_method = st.sidebar.selectbox("Metode Binarisasi Utama (P12):", ["otsu", "adaptive"])

show_pcd_steps = st.sidebar.checkbox("Tampilkan Box Evaluasi Langkah PCD (10 Tahap)", value=True)
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
    detector, recognizer, preprocessor = load_models()
    binarizer = PlateBinarizer(method=binarize_method)
    detector.conf_threshold = conf_thresh

    if st.button("🚀 Proses Deteksi, Klasifikasi Kendaraan & Evaluasi Pajak", type="primary"):
        with st.spinner("Sedang memproses citra melalui 11 tahap pipeline & evaluasi kebaruan..."):
            # 1. Tahap Deteksi (YOLOv8n + Deskew + Polarity/Color)
            detections = detector.detect(image_bgr)

            # Fallback jika user mengunggah foto pelat yang sudah ter-crop dekat (YOLO tidak mendeteksi mobil)
            if not detections:
                ih, iw = image_bgr.shape[:2]
                ar = round(float(iw) / float(max(1, ih)), 2)
                if 1.4 <= ar <= 5.5:
                    st.info("ℹ️ Gambar terdeteksi sebagai citra pelat jarak dekat (close-up). Memproses seluruh area gambar sebagai ROI pelat.")
                    detections = [{
                        "box": (0, 0, iw, ih),
                        "confidence": 1.0,
                        "class_id": 0,
                        "crop": image_bgr.copy(),
                        "aspect_ratio": ar,
                        "is_complete": True,
                        "status": "Close-up Pelat Lengkap"
                    }]

            if not detections:
                st.warning("⚠️ Tidak ditemukan pelat nomor pada gambar ini. Coba turunkan nilai Confidence Threshold di sidebar.")
                st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Gambar Asli", use_container_width=True)
            else:
                pcd_results = []

                for i, det in enumerate(detections):
                    crop_img = det["crop"]
                    ch, cw = crop_img.shape[:2]

                    # --- Tahap 2: Deskewing (dari detector.py Zulfan atau fallback) ---
                    deskewed_img = det.get("deskewed", crop_img)
                    skew_angle = det.get("skew_angle", 0.0)
                    deskew_status = det.get("deskew_status", f"Sudut: {skew_angle}°")

                    # --- Tahap 3: Kebaruan 1 (Warna Pelat, Jenis Kendaraan & Standar Polaritas) ---
                    if "polarity_std" in det and "plate_color" in det:
                        polarity_std_img = det["polarity_std"]
                        plate_color = det["plate_color"]
                        vehicle_type = det.get("vehicle_type", f"Kendaraan ({plate_color})")
                        polarity_status = det.get("polarity_status", "Standar Latar Hitam & Teks Putih")
                        hsv_summary = det.get("hsv_summary", "-")
                    else:
                        fb_pol = fallback_color_and_polarity(deskewed_img)
                        polarity_std_img = fb_pol["polarity_std"]
                        plate_color = fb_pol["plate_color"]
                        vehicle_type = fb_pol["vehicle_type"]
                        polarity_status = fb_pol["polarity_status"]
                        hsv_summary = fb_pol["hsv_summary"]

                    if save_to_output:
                        save_step_output(crop_img, os.path.join(OUTPUT_DIR, "01_cropped_plates"), f"crop_{i+1}_{image_name}")

                    # --- Tahap 4, 5, 6: Preprocessing (Grayscale+Smooth, CLAHE Conditional, Unsharp Masking) ---
                    prep_res = preprocessor.process(polarity_std_img)
                    gray_img = prep_res["grayscale"]
                    smoothed_img = prep_res.get("smoothed", gray_img)
                    clahe_img = prep_res["enhanced"]

                    if "sharpened" in prep_res:
                        sharpened_img = prep_res["sharpened"]
                        sharpness_score = prep_res.get("sharpness_score", 0.0)
                    else:
                        # Fallback Unsharp Masking jika preprocessor.py belum diupdate Ucup
                        blur_tmp = cv2.GaussianBlur(clahe_img, (0, 0), 2.0)
                        sharpened_img = cv2.addWeighted(clahe_img, 1.6, blur_tmp, -0.6, 0)
                        sharpness_score = round(float(cv2.Laplacian(sharpened_img, cv2.CV_64F).var()), 1)

                    if save_to_output:
                        save_step_output(sharpened_img, os.path.join(OUTPUT_DIR, "02_preprocessed"), f"prep_{i+1}_{image_name}")

                    # --- Tahap 7, 8, 9: Binarisasi, Morfologi & Segmentasi Dual-ROI (Utama vs Pajak) ---
                    bin_res = binarizer.process(sharpened_img)
                    binary_raw = bin_res.get("binary_raw", bin_res["binary_normalized"])
                    morphed_img = bin_res.get("morphed", binary_raw)
                    roi_cleaned = bin_res.get("roi_cleaned", bin_res["binary_normalized"])
                    removed_blobs = bin_res.get("removed_blobs", "-")
                    white_ratio = bin_res.get(
                        "white_ratio",
                        round(float(np.mean(binary_raw == 255) * 100.0), 1)
                    )

                    if "tax_roi_binary" in bin_res and bin_res["tax_roi_binary"] is not None:
                        tax_roi_bin = bin_res["tax_roi_binary"]
                        tax_roi_gray = bin_res.get("tax_roi_gray", None)
                    else:
                        # Fallback ekstraksi ROI Pajak jika binarizer.py belum diupdate Ucup
                        tax_roi_bin, tax_roi_gray = fallback_extract_tax_roi(
                            bin_res.get("upscaled_gray", sharpened_img),
                            roi_cleaned
                        )

                    if save_to_output:
                        save_step_output(roi_cleaned, os.path.join(OUTPUT_DIR, "03_binarized"), f"binary_{i+1}_{image_name}")

                    # --- Tahap 10A: Pembacaan Teks Nomor Utama (Multi-Input OCR Voting + Regex) ---
                    ocr_res = recognizer.recognize(
                        roi_cleaned,
                        fallback_image=bin_res.get("upscaled_gray", sharpened_img),
                        extra_images=bin_res.get("variants", [])[:1]
                    )

                    # --- Tahap 10B: Kebaruan 2 (Pembacaan Masa Berlaku Pajak MM.YY & Status Aktif/Mati) ---
                    tax_res = recognizer.recognize_tax(tax_roi_bin, tax_gray=tax_roi_gray)

                    plate_text = ocr_res["clean_text"]
                    det["text"] = plate_text
                    det["plate_color"] = plate_color
                    det["vehicle_type"] = vehicle_type
                    det["tax_text"] = tax_res["tax_text"]
                    det["tax_status"] = tax_res["tax_status"]
                    det["is_tax_active"] = tax_res["is_tax_active"]
                    det["is_occluded"] = (
                        ocr_res["is_occluded"]
                        or (not det["is_complete"])
                        or (not ocr_res["is_valid_format"])
                        or ocr_res["confidence"] < 0.4
                    )

                    gt_text = get_ground_truth(image_name, BASE_DIR)

                    pcd_results.append({
                        "index": i + 1,
                        "crop": crop_img,
                        "crop_size": f"{cw}x{ch} px",
                        "yolo_conf": round(det["confidence"], 2),
                        "deskewed": deskewed_img,
                        "skew_angle": skew_angle,
                        "deskew_status": deskew_status,
                        "polarity_std": polarity_std_img,
                        "polarity_status": polarity_status,
                        "plate_color": plate_color,
                        "vehicle_type": vehicle_type,
                        "hsv_summary": hsv_summary,
                        "grayscale": gray_img,
                        "smoothed": smoothed_img,
                        "clahe": clahe_img,
                        "clahe_status": prep_res.get("clahe_status", "Aktif"),
                        "hist_mean": prep_res.get("hist_mean", "-"),
                        "hist_std": prep_res.get("hist_std", "-"),
                        "sharpened": sharpened_img,
                        "sharpness_score": sharpness_score,
                        "binary_raw": binary_raw,
                        "white_ratio": white_ratio,
                        "morphed": morphed_img,
                        "roi_cleaned": roi_cleaned,
                        "removed_blobs": removed_blobs,
                        "tax_roi_bin": tax_roi_bin,
                        "tax_roi_gray": tax_roi_gray,
                        "text": plate_text,
                        "confidence": ocr_res["confidence"],
                        "winning_source": ocr_res.get("source", "-"),
                        "all_candidates": ocr_res.get("all_candidates", []),
                        "tax_text": tax_res["tax_text"],
                        "tax_status": tax_res["tax_status"],
                        "is_tax_active": tax_res["is_tax_active"],
                        "tax_confidence": tax_res["tax_confidence"],
                        "aspect_ratio": det["aspect_ratio"],
                        "is_complete": det["is_complete"],
                        "status": det["status"],
                        "is_occluded": det["is_occluded"],
                        "ground_truth": gt_text,
                    })

                # 11. Gambar Anotasi Visual (Bounding Box + Leader Line + Badge 2 Baris)
                annotated_img = draw_plate_annotations(image_bgr, detections)

                if save_to_output:
                    save_step_output(annotated_img, os.path.join(OUTPUT_DIR, "04_final_detections"), f"final_{image_name}")

                # Tampilkan Hasil Utama
                col_main, col_info = st.columns([1.85, 1.15])

                with col_main:
                    st.subheader("🖼️ Hasil Deteksi & Visualisasi Garis Penunjuk")
                    st.image(
                        cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
                        caption="Gambar Beranotasi (Bounding Box + Garis Penunjuk + Info Warna & Pajak)",
                        use_container_width=True
                    )

                with col_info:
                    st.subheader("📋 Hasil Bacaan & Evaluasi Kebaruan")
                    for res in pcd_results:
                        st.markdown(f"### Pelat #{res['index']}: **`{res['text']}`**")

                        # Status keutuhan plat
                        if res["is_occluded"] or not res["is_complete"]:
                            st.warning(f"⚠️ **Status Pelat: Terhalang / Tidak Utuh** ({res['status']})")
                        else:
                            st.success("✅ **Status Pelat: Utuh & Format Valid**")

                        # KEBARUAN 1: Warna Pelat & Jenis Kendaraan
                        st.info(
                            f"🎨 **Warna Pelat:** `{res['plate_color']}`  \n"
                            f"🚘 **Klasifikasi:** **{res['vehicle_type']}**"
                        )

                        # KEBARUAN 2: Masa Berlaku Pajak & Status Aktif/Mati
                        if res["is_tax_active"] is True:
                            st.success(
                                f"📅 **Masa Berlaku Pajak:** `{res['tax_text']}`  \n"
                                f"🟢 **Status Pajak:** **{res['tax_status']}** (Conf: {res['tax_confidence']})"
                            )
                        elif res["is_tax_active"] is False:
                            st.error(
                                f"📅 **Masa Berlaku Pajak:** `{res['tax_text']}`  \n"
                                f"🔴 **Status Pajak:** **{res['tax_status']}** (Conf: {res['tax_confidence']})"
                            )
                        else:
                            st.warning(
                                f"📅 **Masa Berlaku Pajak:** `{res['tax_text']}`  \n"
                                f"⚪ **Status Pajak:** Tidak Terdeteksi / Tertutup Bingkai"
                            )

                        st.markdown(
                            f"* **Confidence OCR Utama:** `{res['confidence']}` *(Sumber: {res['winning_source']})*\n"
                            f"* **Rasio Aspek (W/H):** `{res['aspect_ratio']}` | **Kemiringan:** `{res['skew_angle']}°`\n"
                            f"* **Analisis Histogram (P6):** {res['clahe_status']}"
                        )

                        # Ground Truth Match
                        if res["ground_truth"]:
                            st.info(f"🎯 **Ground Truth Dataset:** `{res['ground_truth']}`")
                            is_match = (res['text'].replace(" ", "") == res['ground_truth'].replace(" ", ""))
                            if is_match:
                                st.success("🎉 Cocok 100% dengan Kunci Jawaban!")
                            else:
                                st.warning("🔍 Terdapat perbedaan karakter dengan Ground Truth.")

                        # Tabel Komparasi Multi-Input OCR
                        if res["all_candidates"]:
                            st.markdown("**📊 Tabel Komparasi 3 Kandidat Input OCR:**")
                            table_rows = []
                            for c in res["all_candidates"]:
                                table_rows.append({
                                    "Kandidat Input Citra": c.get("source", "-"),
                                    "Hasil Bacaan": c.get("fmt", "-"),
                                    "Confidence": c.get("conf", 0.0),
                                    "Valid Regex": "✅ Ya" if c.get("valid") else "❌ Tidak"
                                })
                            st.dataframe(table_rows, use_container_width=True, hide_index=True)

                        st.divider()

                # Box Evaluasi Langkah Pengolahan Citra Digital (10 Kotak Sesuai Diagram)
                if show_pcd_steps:
                    st.subheader("🔬 Box Evaluasi Tahapan Pengolahan Citra Digital (PCD)")

                    for res in pcd_results:
                        st.markdown(f"#### Detail Box Evaluasi — Pelat #{res['index']} (`{res['text']}`)")

                        # Baris 1: Blok YOLOv8n & Normalisasi (Tahap 1 - 3)
                        st.markdown("**Blok 1: Deteksi YOLOv8n, Koreksi Geometri & Standarisasi Polaritas Warna**")
                        b1, b2, b3 = st.columns(3)
                        with b1:
                            st.markdown("**1. Deteksi & Pemotongan (Crop)**")
                            st.image(cv2.cvtColor(res["crop"], cv2.COLOR_BGR2RGB), use_container_width=True)
                            st.caption(
                                f"YOLO Conf: `{res['yolo_conf']}` | Ukuran: `{res['crop_size']}` | "
                                f"Rasio W/H: `{res['aspect_ratio']}`"
                            )
                        with b2:
                            st.markdown("**2. Koreksi Kemiringan (Deskewing)**")
                            st.image(cv2.cvtColor(res["deskewed"], cv2.COLOR_BGR2RGB), use_container_width=True)
                            st.caption(f"Sudut Kemiringan: `{res['skew_angle']}°` | {res['deskew_status']}")
                        with b3:
                            st.markdown("**3. Standar Polaritas & Warna HSV**")
                            pol_vis = (
                                cv2.cvtColor(res["polarity_std"], cv2.COLOR_BGR2RGB)
                                if res["polarity_std"].ndim == 3 else res["polarity_std"]
                            )
                            st.image(pol_vis, clamp=True, use_container_width=True)
                            st.caption(
                                f"Warna: `{res['plate_color']}` ({res['hsv_summary']}) | {res['polarity_status']}"
                            )

                        # Baris 2: Peningkatan Kualitas Citra Grayscale (Tahap 4 - 6)
                        st.markdown("**Blok 2: Peningkatan Kualitas Citra (Smoothing, Conditional CLAHE & Sharpening)**")
                        b4, b5, b6 = st.columns(3)
                        with b4:
                            st.markdown("**4. Grayscale & Smoothing**")
                            st.image(res["smoothed"], clamp=True, use_container_width=True)
                            st.caption("Konversi Grayscale + Upscaling + Bilateral Filter (d=5, σ=35)")
                        with b5:
                            st.markdown("**5. Peningkatan Kontras (CLAHE Conditional)**")
                            st.image(res["clahe"], clamp=True, use_container_width=True)
                            st.caption(f"{res['clahe_status']} (μ={res['hist_mean']}, σ={res['hist_std']})")
                        with b6:
                            st.markdown("**6. Penajaman Citra (Unsharp Masking)**")
                            st.image(res["sharpened"], clamp=True, use_container_width=True)
                            st.caption(f"Unsharp Masking Aktif | Skor Ketajaman (Laplacian Var): `{res['sharpness_score']}`")

                        # Baris 3: Segmentasi Biner, Morfologi & Pemisahan Dual-ROI (Tahap 7 - 9b)
                        st.markdown("**Blok 3: Segmentasi Biner, Perbaikan Morfologi & Pemisahan Dual-ROI (Nomor Utama vs Pajak)**")
                        b7, b8, b9a, b9b = st.columns(4)
                        with b7:
                            st.markdown("**7. Segmentasi Biner**")
                            st.image(res["binary_raw"], clamp=True, use_container_width=True)
                            st.caption(f"Metode: `{binarize_method.upper()}` | Rasio Piksel Putih: `{res['white_ratio']}%`")
                        with b8:
                            st.markdown("**8. Perbaikan Morfologi**")
                            st.image(res["morphed"], clamp=True, use_container_width=True)
                            st.caption("Morphological Closing (Menyambung celah huruf/angka)")
                        with b9a:
                            st.markdown("**9a. ROI Nomor Utama**")
                            st.image(res["roi_cleaned"], clamp=True, use_container_width=True)
                            st.caption(f"Hasil Bacaan: `{res['text']}` | Noise/Pajak Dibuang: `{res['removed_blobs']}` blob")
                        with b9b:
                            st.markdown("**9b. ROI Nomor Pajak (MM.YY)**")
                            st.image(res["tax_roi_bin"], clamp=True, use_container_width=True)
                            st.caption(f"Pajak Terbaca: `{res['tax_text']}` | Status: **{res['tax_status']}**")

                        st.divider()

                if save_to_output:
                    st.success(f"📁 Seluruh hasil proses telah disimpan ke folder `{OUTPUT_DIR}`!")

else:
    st.info("👈 Silakan unggah gambar kendaraan/pelat nomor atau pilih salah satu sampel uji dari sidebar untuk memulai.")
