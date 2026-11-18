# Aesthetic Archetype: A. Precision Technical Dark (Linear / Raycast / Vercel style)
import os
import glob
import urllib.request
import cv2
import numpy as np
import streamlit as st

from Script.detector import PlateDetector
from Script.utils import draw_plate_annotations, get_ground_truth, save_step_output
from Script.plate_tax_pipeline import (
    analyze_single_plate,
    preprocess_plate,
    process_main_plate_roi,
    process_tax_roi,
)

st.set_page_config(
    page_title="ALPR Indonesia — Vehicle & License Plate Telemetry",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# DTCG DESIGN TOKENS & ANTI-SLOP PRECISION TECHNICAL STYLING
# ==============================================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --bg-canvas: #0A0B0E;
    --bg-surface: #111318;
    --bg-elevated: #171A21;
    --border-hairline: rgba(255, 255, 255, 0.09);
    --border-focus: #3B82F6;
    --text-primary: #F3F4F6;
    --text-secondary: #D1D5DB;
    --text-muted: #9CA3AF;
    --accent-primary: #3B82F6;
    --status-valid-bg: rgba(16, 185, 129, 0.12);
    --status-valid-fg: #10B981;
    --status-warn-bg: rgba(245, 158, 11, 0.14);
    --status-warn-fg: #F59E0B;
    --status-err-bg: rgba(239, 68, 68, 0.14);
    --status-err-fg: #EF4444;
    --radius-sm: 6px;
    --radius-md: 8px;
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
}

html, body, [class*="css"] {
    font-family: var(--font-sans);
    color: var(--text-primary);
}

.alpr-header {
    border-bottom: 1px solid var(--border-hairline);
    padding-bottom: 1rem;
    margin-bottom: 1.5rem;
}

.alpr-eyebrow {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--accent-primary);
    margin-bottom: 0.35rem;
}

.alpr-title {
    font-size: 1.65rem;
    font-weight: 700;
    letter-spacing: -0.025em;
    color: var(--text-primary);
    margin: 0 0 0.35rem 0;
}

.alpr-subtitle {
    font-size: 0.9rem;
    color: var(--text-muted);
    margin: 0;
}

.alpr-panel {
    background-color: var(--bg-surface);
    border: 1px solid var(--border-hairline);
    border-radius: var(--radius-md);
    padding: 1.15rem 1.25rem;
    margin-bottom: 1rem;
}

.alpr-panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid var(--border-hairline);
    padding-bottom: 0.65rem;
    margin-bottom: 0.9rem;
}

.alpr-section-tag {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-muted);
}

.alpr-metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 0.85rem;
    margin-top: 0.5rem;
}

.alpr-card {
    background-color: var(--bg-elevated);
    border: 1px solid var(--border-hairline);
    border-radius: var(--radius-sm);
    padding: 0.95rem 1rem;
}

.alpr-card-label {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.4rem;
}

.alpr-card-value {
    font-family: var(--font-mono);
    font-size: 1.45rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    font-variant-numeric: tabular-nums;
    color: var(--text-primary);
    margin-bottom: 0.45rem;
}

.alpr-card-meta {
    font-size: 0.78rem;
    color: var(--text-muted);
    font-variant-numeric: tabular-nums;
}

.alpr-badge {
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 0.18rem 0.5rem;
    border-radius: 4px;
    font-variant-numeric: tabular-nums;
}

.badge-valid {
    background-color: var(--status-valid-bg);
    color: var(--status-valid-fg);
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.badge-warn {
    background-color: var(--status-warn-bg);
    color: var(--status-warn-fg);
    border: 1px solid rgba(245, 158, 11, 0.3);
}

.badge-err {
    background-color: var(--status-err-bg);
    color: var(--status-err-fg);
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.alpr-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.84rem;
    font-variant-numeric: tabular-nums;
}

.alpr-table th, .alpr-table td {
    text-align: left;
    padding: 0.55rem 0.75rem;
    border-bottom: 1px solid var(--border-hairline);
}

.alpr-table th {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
}

.alpr-table td.mono {
    font-family: var(--font-mono);
}
</style>
""",
    unsafe_allow_html=True,
)

# ==============================================================================
# INISIALISASI PATH & CACHING MODEL
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "Output")


def _discover_sample_dirs():
    dirs = {}
    kaggle_base = os.path.expanduser(
        "~/.cache/kagglehub/datasets/juanthomaswijaya/indonesian-license-plate-dataset/versions/1"
    )
    det_candidates = [
        os.path.join(BASE_DIR, "Dataset", "Raw", "Indonesian License Plate Dataset", "images", "test"),
        os.path.join(kaggle_base, "Indonesian License Plate Dataset", "images", "test"),
    ]
    rec_candidates = [
        os.path.join(kaggle_base, "Indonesian License Plate Recognition Dataset", "images", "test"),
        os.path.join(kaggle_base, "Indonesian License Plate Recognition Dataset", "images", "train"),
    ]
    for d in det_candidates:
        if os.path.exists(d):
            dirs["Kendaraan Utuh (Detection Test Set)"] = d
            break
    for d in rec_candidates:
        if os.path.exists(d):
            label = "Potongan Plat (Recognition Test Set)" if d.endswith("test") else "Potongan Plat (Recognition Train Set)"
            dirs[label] = d
    return dirs


@st.cache_resource
def load_pipeline_models():
    import easyocr
    import torch

    use_gpu = torch.cuda.is_available()
    detector = PlateDetector()
    ocr_reader = easyocr.Reader(["en"], gpu=use_gpu)
    return detector, ocr_reader


def fetch_image_from_url(url):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) ALPR-Client/1.0"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ==============================================================================
# HEADER UTAMA
# ==============================================================================
st.markdown(
    """
<div class="alpr-header">
    <div class="alpr-eyebrow">SISTEM VISI KOMPUTER &amp; PENGOLAHAN CITRA DIGITAL</div>
    <h1 class="alpr-title">Deteksi &amp; Pengenalan Pelat Nomor Kendaraan Indonesia</h1>
    <p class="alpr-subtitle">
        Lokalisasi YOLOv8, Koreksi Kemiringan (Deskewing), Klasifikasi Warna HSV, serta Ekstraksi Nomor Plat &amp; Masa Berlaku Pajak (MM-YY).
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# ==============================================================================
# SIDEBAR: KONFIGURASI PARAMETER
# ==============================================================================
st.sidebar.markdown("### `[CFG]` Parameter Pipeline")
conf_thresh = st.sidebar.slider("Confidence Threshold YOLO", 0.10, 0.90, 0.25, 0.05)
use_vlm = st.sidebar.checkbox("Aktifkan Verifikasi Konsensus AI Lokal (Qwen 3.5)", value=True)
allow_direct_plate = st.sidebar.checkbox(
    "Fallback Deteksi Langsung (Jika Input Berupa Crop Plat)", value=True
)
show_pcd_steps = st.sidebar.checkbox("Tampilkan Tahapan Pengolahan Citra (PCD)", value=True)
save_to_output = st.sidebar.checkbox("Simpan Hasil ke Folder Output/", value=False)

# ==============================================================================
# PANEL INPUT MULTI-SUMBER (UPLOAD / DRAG-AND-DROP, URL, DATASET)
# ==============================================================================
tab_upload, tab_url, tab_dataset = st.tabs(
    ["[01] Unggah File / Drag & Drop", "[02] Impor dari URL Gambar", "[03] Sampel Dataset"]
)

image_bgr = None
image_name = "input_vehicle.jpg"

with tab_upload:
    uploaded_file = st.file_uploader(
        "Tarik dan lepaskan foto kendaraan atau pelat nomor (JPG, JPEG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
    )
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_name = uploaded_file.name

with tab_url:
    col_u1, col_u2 = st.columns([4, 1])
    with col_u1:
        img_url = st.text_input(
            "Masukkan URL Gambar Langsung (http/https)",
            placeholder="https://example.com/foto-kendaraan-plat-indonesia.jpg",
        )
    with col_u2:
        st.write("")
        st.write("")
        load_url_btn = st.button("Muat URL", use_container_width=True)
    if img_url and (load_url_btn or image_bgr is None):
        try:
            loaded = fetch_image_from_url(img_url.strip())
            if loaded is not None:
                image_bgr = loaded
                image_name = os.path.basename(img_url.split("?")[0]) or "url_image.jpg"
            else:
                st.error("Gagal mendekode gambar dari URL tersebut.")
        except Exception as exc:
            st.error(f"Gagal mengunduh URL: {exc}")

with tab_dataset:
    sample_dirs = _discover_sample_dirs()
    if sample_dirs:
        col_d1, col_d2 = st.columns([1, 2])
        with col_d1:
            selected_group = st.selectbox("Pilih Kategori Dataset", list(sample_dirs.keys()))
        target_dir = sample_dirs[selected_group]
        files = sorted(
            [f for f in os.listdir(target_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        )
        with col_d2:
            selected_file = st.selectbox("Pilih File Citra Uji", ["(Pilih sampel...)"] + files)
        if selected_file != "(Pilih sampel...)" and image_bgr is None:
            sample_path = os.path.join(target_dir, selected_file)
            image_bgr = cv2.imread(sample_path)
            image_name = selected_file
    else:
        st.info("Dataset lokal belum ditemukan di cache. Gunakan tab Unggah File atau URL.")

# ==============================================================================
# EKSEKUSI PIPELINE & TAMPILAN UTAMA
# ==============================================================================
if image_bgr is None:
    st.markdown(
        """
    <div class="alpr-panel" style="text-align: center; padding: 2.5rem 1.5rem;">
        <div class="alpr-section-tag">[STATUS: MENUNGGU INPUT CITRA]</div>
        <p style="color: var(--text-secondary); margin: 0.6rem 0 0.25rem 0; font-weight: 500;">
            Belum ada citra yang dipilih untuk dianalisis.
        </p>
        <p style="color: var(--text-muted); font-size: 0.85rem; margin: 0;">
            Unggah foto kendaraan dari perangkat Anda, tempel tautan URL gambar, atau pilih sampel dari tab di atas.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )
else:
    detector, ocr_reader = load_pipeline_models()
    detector.conf_threshold = conf_thresh

    with st.spinner("Menjalankan deteksi YOLOv8, analisis warna HSV, dan pengenalan karakter..."):
        detections = detector.detect(image_bgr) if detector.is_plate_model else []

        # Jika tidak ada deteksi YOLO dan opsi fallback aktif, gunakan seluruh citra sebagai ROI plat
        used_direct_fallback = False
        if not detections and allow_direct_plate:
            ih, iw = image_bgr.shape[:2]
            ar, is_comp, st_desc = detector.check_aspect_ratio(iw, ih)
            crop_info = detector.process_crop(image_bgr)
            det_item = {
                "box": (4, 4, max(5, iw - 4), max(5, ih - 4)),
                "confidence": 1.0,
                "class_id": 0,
                "crop": image_bgr.copy(),
                "aspect_ratio": round(ar, 2),
                "is_complete": is_comp,
                "status": f"Input Langsung ({st_desc})",
            }
            det_item.update(crop_info)
            detections = [det_item]
            used_direct_fallback = True

        enriched_detections = []
        for idx, det in enumerate(detections):
            crop_bgr = det["crop"]
            deskewed_bgr = det["deskewed"]

            if save_to_output:
                save_step_output(
                    crop_bgr,
                    os.path.join(OUTPUT_DIR, "01_cropped_plates"),
                    f"crop_{idx + 1}_{image_name}",
                )

            # Jalankan pipeline pengenalan nomor plat & angka pajak
            ocr_res = analyze_single_plate(deskewed_bgr, ocr_reader, use_vlm=use_vlm)
            prep_vis = preprocess_plate(deskewed_bgr)
            main_vars, _ = process_main_plate_roi(
                prep_vis["enhanced"], prep_vis["yellow_channel"], raw_gray=prep_vis["gray"]
            )
            tax_vars, _ = process_tax_roi(prep_vis["enhanced"], prep_vis["yellow_channel"])

            if save_to_output:
                save_step_output(
                    prep_vis["enhanced"],
                    os.path.join(OUTPUT_DIR, "02_preprocessed"),
                    f"enhanced_{idx + 1}_{image_name}",
                )
                save_step_output(
                    main_vars["Otsu"],
                    os.path.join(OUTPUT_DIR, "03_binarized"),
                    f"bin_{idx + 1}_{image_name}",
                )

            plate_str = ocr_res["plate_text"]
            tax_str = ocr_res["tax_text"]
            badge_label = plate_str if tax_str == "-" else f"{plate_str} | {tax_str}"

            det["text"] = badge_label
            det["plate_text"] = plate_str
            det["plate_valid"] = ocr_res["plate_valid"]
            det["plate_conf"] = ocr_res["plate_conf"]
            det["plate_status"] = ocr_res["plate_status"]
            det["tax_text"] = tax_str
            det["tax_valid"] = ocr_res["tax_valid"]
            det["tax_conf"] = ocr_res["tax_conf"]
            det["tax_status"] = ocr_res["tax_status"]
            det["sharpness_lap"] = ocr_res["sharpness_lap"]
            det["is_occluded"] = not ocr_res["plate_valid"]
            det["prep_vis"] = prep_vis
            det["main_vars"] = main_vars
            det["tax_vars"] = tax_vars
            enriched_detections.append(det)

        annotated_bgr = (
            draw_plate_annotations(image_bgr, enriched_detections)
            if enriched_detections
            else image_bgr.copy()
        )
        if save_to_output and enriched_detections:
            save_step_output(
                annotated_bgr,
                os.path.join(OUTPUT_DIR, "04_final_detections"),
                f"final_{image_name}",
            )

    # ==========================================================================
    # BAGIAN ATAS: CITRA INPUT VS CITRA OUTPUT DENGAN BOUNDING BOX
    # ==========================================================================
    img_h, img_w = image_bgr.shape[:2]
    col_in, col_out = st.columns(2)

    with col_in:
        st.markdown(
            f"""
        <div class="alpr-panel">
            <div class="alpr-panel-header">
                <span class="alpr-section-tag">[INPUT] CITRA SUMBER</span>
                <span class="alpr-section-tag">{image_name} ({img_w}x{img_h} px)</span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col_out:
        mode_tag = "DIRECT ROI MODE" if used_direct_fallback else f"YOLOV8 ({len(enriched_detections)} OBJEK)"
        st.markdown(
            f"""
        <div class="alpr-panel">
            <div class="alpr-panel-header">
                <span class="alpr-section-tag">[OUTPUT] LOKALISASI BOUNDING BOX &amp; ANOTASI</span>
                <span class="alpr-section-tag">{mode_tag}</span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        st.image(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

        ok_enc, buf_enc = cv2.imencode(".jpg", annotated_bgr)
        if ok_enc:
            st.download_button(
                label="Unduh Gambar Hasil Deteksi (.JPG)",
                data=buf_enc.tobytes(),
                file_name=f"detected_{image_name}",
                mime="image/jpeg",
            )

    # ==========================================================================
    # BAGIAN BAWAH: PANEL INFORMASI TERSTRUKTUR PER PELAT NOMOR
    # ==========================================================================
    st.markdown("---")
    st.markdown(
        """
    <div style="margin-bottom: 0.75rem;">
        <div class="alpr-eyebrow">TELEMETRI &amp; INSPEKSI DETAIL</div>
        <h2 style="font-size: 1.25rem; font-weight: 600; margin: 0;">Panel Informasi Hasil Ekstraksi Pelat Nomor</h2>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if not enriched_detections:
        st.warning(
            "Tidak ditemukan objek pelat nomor pada citra ini. Coba turunkan Confidence Threshold di sidebar atau aktifkan Fallback Deteksi Langsung."
        )
    else:
        gt_val = get_ground_truth(image_name, BASE_DIR)

        for i, det in enumerate(enriched_detections, start=1):
            p_valid = det["plate_valid"]
            t_valid = det["tax_valid"]

            p_badge_cls = "badge-valid" if p_valid else "badge-err"
            p_badge_txt = "VALID" if p_valid else "TIDAK VALID"

            t_badge_cls = "badge-valid" if t_valid else "badge-warn"
            t_badge_txt = "VALID (MM-YY)" if t_valid else "TIDAK VALID"

            c_badge_cls = "badge-valid" if det["is_complete"] else "badge-warn"
            c_badge_txt = "UTUH" if det["is_complete"] else "TERPOTONG / MIRING"

            gt_html = (
                f'<span class="alpr-section-tag">GROUND TRUTH: {gt_val}</span>'
                if gt_val
                else f'<span class="alpr-section-tag">CONFIDENCE YOLO: {det["confidence"]:.0%}</span>'
            )

            st.markdown(
                f"""
            <div class="alpr-panel">
                <div class="alpr-panel-header">
                    <span class="alpr-section-tag">[PELAT #{i:02d}] KOORDINAT BBOX: {det['box']}</span>
                    {gt_html}
                </div>
                <div class="alpr-metric-grid">
                    <div class="alpr-card">
                        <div class="alpr-card-label">[01] Nomor Plat Utama</div>
                        <div class="alpr-card-value">{det['plate_text']}</div>
                        <div>
                            <span class="alpr-badge {p_badge_cls}">{p_badge_txt}</span>
                            <span class="alpr-card-meta" style="margin-left: 0.4rem;">
                                Conf: {det['plate_conf']:.0%}
                            </span>
                        </div>
                        <div class="alpr-card-meta" style="margin-top: 0.35rem;">
                            Sumber: {det['plate_status']}
                        </div>
                    </div>
                    <div class="alpr-card">
                        <div class="alpr-card-label">[02] Angka Pajak (MM-YY)</div>
                        <div class="alpr-card-value">{det['tax_text']}</div>
                        <div>
                            <span class="alpr-badge {t_badge_cls}">{t_badge_txt}</span>
                            <span class="alpr-card-meta" style="margin-left: 0.4rem;">
                                Conf: {det['tax_conf']:.0%}
                            </span>
                        </div>
                        <div class="alpr-card-meta" style="margin-top: 0.35rem;">
                            Status: {det['tax_status']}
                        </div>
                    </div>
                    <div class="alpr-card">
                        <div class="alpr-card-label">[03] Warna Pelat &amp; Golongan</div>
                        <div class="alpr-card-value">{det['plate_color']}</div>
                        <div>
                            <span class="alpr-badge badge-valid">{det['vehicle_type']}</span>
                        </div>
                        <div class="alpr-card-meta" style="margin-top: 0.35rem;">
                            {det['hsv_summary']}
                        </div>
                    </div>
                    <div class="alpr-card">
                        <div class="alpr-card-label">[04] Geometri &amp; Kualitas Citra</div>
                        <div class="alpr-card-value">{det['aspect_ratio']:.2f} : 1</div>
                        <div>
                            <span class="alpr-badge {c_badge_cls}">{c_badge_txt}</span>
                            <span class="alpr-card-meta" style="margin-left: 0.4rem;">
                                Sharpness: {det['sharpness_lap']}
                            </span>
                        </div>
                        <div class="alpr-card-meta" style="margin-top: 0.35rem;">
                            Deskew: {det['deskew_status']}
                        </div>
                    </div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            if show_pcd_steps:
                with st.expander(
                    f"[PCD PIPELINE #{i:02d}] Inspeksi Visual Tahapan Pengolahan Citra Digital & ROI",
                    expanded=(i == 1),
                ):
                    c1, c2, c3, c4, c5 = st.columns(5)
                    with c1:
                        st.caption("01. Crop ROI Awal")
                        st.image(
                            cv2.cvtColor(det["crop"], cv2.COLOR_BGR2RGB),
                            use_container_width=True,
                        )
                    with c2:
                        st.caption(f"02. Deskew ({det['skew_angle']:+.1f} deg)")
                        st.image(
                            cv2.cvtColor(det["deskewed"], cv2.COLOR_BGR2RGB),
                            use_container_width=True,
                        )
                    with c3:
                        st.caption("03. Grayscale & CLAHE")
                        st.image(det["prep_vis"]["enhanced"], clamp=True, use_container_width=True)
                    with c4:
                        st.caption("04. Biner ROI Plat Utama")
                        st.image(det["main_vars"]["Otsu"], clamp=True, use_container_width=True)
                    with c5:
                        st.caption("05. ROI Angka Pajak (Bawah)")
                        st.image(det["tax_vars"]["Sharp"], clamp=True, use_container_width=True)
