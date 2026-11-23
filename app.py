# Aesthetic Archetype: A. Precision Technical Dark (HeyPuter AI Builder / Linear / v0 Split-Pane Studio)
import glob
import hashlib
import os
import textwrap
import urllib.request
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from Script.binarizer import PlateBinarizer
from Script.detector import PlateDetector
from Script.preprocessor import PlatePreprocessor
from Script.custom_ocr import CustomPlateRecognizer
from Script.utils import draw_plate_annotations, get_ground_truth

st.set_page_config(
    page_title="ALPR Studio Indonesia",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# DTCG DESIGN TOKENS — DARK PRECISION AI-BUILDER STUDIO
# ==============================================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --bg-base: #08090a;
    --bg-surface: #0f1011;
    --bg-panel: #131416;
    --bg-elevated: #18181f;
    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-strong: rgba(255, 255, 255, 0.14);
    --text-primary: #f7f8f8;
    --text-secondary: #c4c8d0;
    --text-muted: #8a8f98;
    --accent-primary: #3b82f6;
    --accent-soft: rgba(59, 130, 246, 0.12);
    --status-ok-bg: rgba(16, 185, 129, 0.12);
    --status-ok-fg: #10b981;
    --status-ok-border: rgba(16, 185, 129, 0.30);
    --status-warn-bg: rgba(245, 158, 11, 0.12);
    --status-warn-fg: #f59e0b;
    --status-warn-border: rgba(245, 158, 11, 0.30);
    --status-err-bg: rgba(239, 68, 68, 0.12);
    --status-err-fg: #ef4444;
    --status-err-border: rgba(239, 68, 68, 0.30);
    --radius-sm: 4px;
    --radius-md: 6px;
    --radius-lg: 8px;
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, monospace;
}

.stApp {
    background-color: var(--bg-base);
    color: var(--text-primary);
    font-family: var(--font-sans);
}

[data-testid="stSidebar"] {
    background-color: var(--bg-surface);
    border-right: 1px solid var(--border-subtle);
}

[data-testid="stSidebar"] * {
    color: var(--text-secondary);
}

#MainMenu, footer {visibility: hidden;}

.studio-cmdbar {
    background-color: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 0.9rem 1.25rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.75rem;
}

.studio-title {
    font-size: 1.15rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: var(--text-primary);
    margin: 0;
}

.studio-sub {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin: 0.15rem 0 0 0;
}

.studio-pills {
    display: flex;
    gap: 0.45rem;
    flex-wrap: wrap;
    align-items: center;
}

.studio-pill {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    font-weight: 500;
    padding: 0.22rem 0.55rem;
    border-radius: var(--radius-sm);
    background-color: var(--bg-elevated);
    color: var(--text-secondary);
    border: 1px solid var(--border-subtle);
    font-variant-numeric: tabular-nums;
}

.studio-pill-accent {
    background-color: var(--accent-soft);
    color: var(--accent-primary);
    border-color: rgba(59, 130, 246, 0.30);
}

.pane-box {
    background-color: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 0.95rem 1.1rem;
    margin-bottom: 0.85rem;
}

.pane-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 0.65rem;
    margin-bottom: 0.8rem;
    border-bottom: 1px solid var(--border-subtle);
}

.pane-label {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
}

.pane-meta {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--text-muted);
    font-variant-numeric: tabular-nums;
}

.inspector-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 0.75rem;
}

.plate-replica {
    border-radius: var(--radius-md);
    padding: 0.65rem 1rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    font-family: var(--font-mono);
    text-align: center;
    border: 2px solid var(--border-strong);
    margin-bottom: 0.75rem;
}

.plate-theme-Hitam { background: #111318; color: #f7f8f8; border-color: #3f4451; }
.plate-theme-Putih { background: #f8fafc; color: #0f172a; border-color: #cbd5e1; }
.plate-theme-Kuning { background: #facc15; color: #111827; border-color: #ca8a04; }
.plate-theme-Merah { background: #b91c1c; color: #ffffff; border-color: #ef4444; }

.plate-main-txt {
    font-size: 1.55rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    line-height: 1.15;
    font-variant-numeric: tabular-nums;
}

.plate-tax-txt {
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    opacity: 0.85;
    margin-top: 0.25rem;
    font-variant-numeric: tabular-nums;
}

.spec-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
}

.spec-table td {
    padding: 0.45rem 0;
    border-bottom: 1px solid var(--border-subtle);
    vertical-align: middle;
}

.spec-table tr:last-child td {
    border-bottom: none;
}

.spec-key {
    color: var(--text-muted);
    width: 42%;
}

.spec-val {
    color: var(--text-primary);
    font-weight: 500;
    text-align: right;
}

.spec-mono {
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
}

.badge {
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 0.68rem;
    font-weight: 600;
    padding: 0.16rem 0.5rem;
    border-radius: var(--radius-sm);
    border: 1px solid transparent;
    font-variant-numeric: tabular-nums;
}

.badge-ok { background: var(--status-ok-bg); color: var(--status-ok-fg); border-color: var(--status-ok-border); }
.badge-warn { background: var(--status-warn-bg); color: var(--status-warn-fg); border-color: var(--status-warn-border); }
.badge-err { background: var(--status-err-bg); color: var(--status-err-fg); border-color: var(--status-err-border); }

.empty-state {
    background-color: var(--bg-surface);
    border: 1px dashed var(--border-strong);
    border-radius: var(--radius-lg);
    padding: 2.5rem 1.5rem;
    text-align: center;
    color: var(--text-muted);
}

.empty-state-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.35rem;
}

.stage-index {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--accent-primary);
    font-weight: 600;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_pipeline_modules():
    """Memuat seluruh modul pipeline 11-tahap sekali saja (otomatis memakai GPU CUDA jika tersedia)."""
    detector = PlateDetector(use_gpu=None)
    preprocessor = PlatePreprocessor()
    binarizer = PlateBinarizer()
    ocr_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "ocr_best.pt")
    recognizer = CustomPlateRecognizer(model_path=ocr_model_path, use_gpu=None)
    return detector, preprocessor, binarizer, recognizer


def find_kaggle_samples():
    cache_test = os.path.expanduser(
        "~/.cache/kagglehub/datasets/juanthomaswijaya/indonesian-license-plate-dataset/versions/1/Indonesian License Plate Dataset/images/test"
    )
    if os.path.exists(cache_test):
        return sorted(glob.glob(os.path.join(cache_test, "*.jpg")))[:15]
    return []


def decode_uploaded_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


@st.cache_data(show_spinner=False)
def decode_url_image(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=12) as resp:
        arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def run_full_pipeline(image_bgr, detector, preprocessor, binarizer, recognizer, allow_direct_plate):
    """Menjalankan pipeline 11 tahap sesuai kontrak modul dokumen."""
    raw_detections = detector.detect(image_bgr) if detector.is_plate_model else []
    used_direct = False

    if not raw_detections and allow_direct_plate:
        ih, iw = image_bgr.shape[:2]
        ar, is_comp, st_desc = detector.check_aspect_ratio(iw, ih)
        det_item = {
            "box": (4, 4, max(5, iw - 4), max(5, ih - 4)),
            "confidence": 1.0,
            "class_id": 0,
            "crop": image_bgr.copy(),
            "aspect_ratio": round(ar, 2),
            "is_complete": is_comp,
            "status": f"Input Pelat Langsung ({st_desc})",
        }
        det_item.update(detector.process_crop(image_bgr))
        raw_detections = [det_item]
        used_direct = True

    enriched = []
    for det in raw_detections:
        pre_res = preprocessor.process(det["polarity_std"])
        bin_res = binarizer.process(pre_res["sharpened"])

        ocr_main = recognizer.recognize(
            det["deskewed"],
            fallback_image=bin_res["upscaled_gray"],
            extra_images=[bin_res["roi_cleaned"]] + bin_res["variants"],
        )
        ocr_tax = recognizer.recognize_tax(
            bin_res["tax_roi_binary"],
            tax_gray=bin_res["tax_roi_gray"],
        )

        det["pre"] = pre_res
        det["bin"] = bin_res
        det["plate_text"] = ocr_main["clean_text"]
        det["plate_conf"] = ocr_main["confidence"]
        det["plate_valid"] = ocr_main["is_valid_format"]
        det["ocr_source"] = ocr_main["source"]
        det["all_candidates"] = ocr_main["all_candidates"]

        det["tax_text"] = ocr_tax["tax_text"]
        det["tax_status"] = ocr_tax["tax_status"]
        det["is_tax_active"] = ocr_tax["is_tax_active"]
        det["tax_conf"] = ocr_tax["tax_confidence"]
        det["is_occluded"] = not det["is_complete"]
        enriched.append(det)

    return enriched, used_direct


# ==============================================================================
# SIDEBAR: KONTROL WORKSPACE & INPUT
# ==============================================================================
with st.sidebar:
    st.markdown("### Source & Pipeline")
    input_mode = st.radio(
        "Sumber Citra",
        ["Upload Gambar", "Sampel Cache Kaggle", "URL Gambar"],
        index=0,
    )

    image_bgr = None
    image_name = "input.jpg"

    if input_mode == "Upload Gambar":
        uploaded = st.file_uploader("Pilih foto kendaraan", type=["jpg", "jpeg", "png", "webp"])
        if uploaded is not None:
            image_bgr = decode_uploaded_image(uploaded)
            image_name = uploaded.name
    elif input_mode == "Sampel Cache Kaggle":
        samples = find_kaggle_samples()
        if samples:
            chosen = st.selectbox("Pilih sampel", samples, format_func=os.path.basename)
            image_bgr = cv2.imread(chosen)
            image_name = os.path.basename(chosen)
        else:
            st.info("Cache sampel Kaggle tidak ditemukan.")
    else:
        url_input = st.text_input("URL Gambar")
        if url_input.strip():
            try:
                image_bgr = decode_url_image(url_input.strip())
                image_name = os.path.basename(url_input.strip().split("?")[0]) or "url_image.jpg"
            except Exception as exc:
                st.error(f"Gagal memuat URL: {exc}")

    st.divider()
    st.markdown("### Opsi Inspeksi")
    hide_false_positives = st.checkbox(
        "Saring Deteksi Palsu",
        value=True,
        help="Sembunyikan kandidat tidak valid jika minimal satu pelat valid ditemukan.",
    )
    allow_direct_plate = st.checkbox(
        "Mode Potongan Pelat Langsung",
        value=True,
        help="Proses seluruh gambar sebagai pelat jika YOLO tidak menemukan bounding box.",
    )
    show_pcd_grid = st.checkbox("Tampilkan Grid Evaluasi 10 Tahap PCD", value=True)


# ==============================================================================
# TOP COMMAND BAR
# ==============================================================================
dim_label = f"{image_bgr.shape[1]}x{image_bgr.shape[0]} px" if image_bgr is not None else "Menunggu Input"
st.markdown(
    f"""
<div class="studio-cmdbar">
    <div>
        <h1 class="studio-title">ALPR Studio Indonesia</h1>
        <p class="studio-sub">Deteksi Pelat YOLOv8s &middot; Klasifikasi Warna HSV &middot; Dual-ROI Pajak Aktif/Mati &middot; Multi-Input OCR</p>
    </div>
    <div class="studio-pills">
        <span class="studio-pill studio-pill-accent">AUTO-THRESHOLD</span>
        <span class="studio-pill">11-STAGE PIPELINE</span>
        <span class="studio-pill">{dim_label}</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ==============================================================================
# MAIN WORKSPACE (SPLIT-PANE: CANVAS + CONTEXTUAL INSPECTOR)
# ==============================================================================
if image_bgr is None:
    st.markdown(
        """
    <div class="empty-state">
        <div class="empty-state-title">Workspace Siap</div>
        <div>Unggah foto kendaraan melalui bilah samping untuk memulai deteksi pelat, identifikasi golongan warna HSV, dan validasi status pajak.</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
else:
    detector, preprocessor, binarizer, recognizer = load_pipeline_modules()

    image_hash = hashlib.sha256(image_bgr.tobytes()).hexdigest()
    cache_key = (image_hash, bool(allow_direct_plate))

    if st.session_state.get("pipeline_cache_key") != cache_key:
        with st.spinner("Menjalankan pipeline 11 tahap (YOLO -> Deskew -> HSV -> PCD -> Dual-ROI OCR)..."):
            all_candidates, used_direct = run_full_pipeline(
                image_bgr, detector, preprocessor, binarizer, recognizer, allow_direct_plate
            )
        st.session_state["pipeline_cache_key"] = cache_key
        st.session_state["pipeline_all_candidates"] = all_candidates
        st.session_state["pipeline_used_direct"] = used_direct
        st.session_state["pipeline_gt_val"] = get_ground_truth(image_name)
        st.session_state["plate_inspector_idx"] = 0
        st.session_state.pop("annot_cache_key", None)
        st.session_state.pop("annot_cache_img", None)

    all_candidates = st.session_state["pipeline_all_candidates"]
    used_direct = st.session_state["pipeline_used_direct"]
    gt_val = st.session_state["pipeline_gt_val"]

    has_valid = any(d["plate_valid"] for d in all_candidates)
    if hide_false_positives and has_valid:
        detections = [d for d in all_candidates if d["plate_valid"]]
    else:
        detections = all_candidates

    total_det = len(detections)
    valid_det = sum(1 for d in detections if d["plate_valid"])
    active_tax = sum(1 for d in detections if d["is_tax_active"] is True)
    avg_conf = (sum(d["plate_conf"] for d in detections) / total_det) if total_det else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Pelat Terdeteksi", f"{total_det}")
    k2.metric("Format Valid", f"{valid_det}/{total_det}")
    k3.metric("Pajak Aktif", f"{active_tax}/{total_det}")
    k4.metric("Rata-rata Confidence", f"{avg_conf:.0%}")

    left_pane, right_pane = st.columns([1.25, 1.0], gap="medium")

    with left_pane:
        mode_tag = "DIRECT FALLBACK" if used_direct else f"YOLO: {len(detections)} TERDETEKSI"
        st.markdown(
            textwrap.dedent(
                f"""
                <div class="pane-box">
                <div class="pane-header">
                <span class="pane-label">Canvas / Visual Localization</span>
                <span class="pane-meta">{image_name} &middot; {mode_tag}</span>
                </div>
                </div>
                """
            ).strip(),
            unsafe_allow_html=True,
        )
        tab_annot, tab_orig = st.tabs(["Hasil Deteksi & Anotasi", "Citra Asli"])
        with tab_annot:
            annot_key = (cache_key, bool(hide_false_positives))
            if st.session_state.get("annot_cache_key") != annot_key:
                st.session_state["annot_cache_img"] = draw_plate_annotations(image_bgr, detections)
                st.session_state["annot_cache_key"] = annot_key
            annotated_bgr = st.session_state["annot_cache_img"]
            st.image(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
        with tab_orig:
            st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

    with right_pane:
        st.markdown(
            textwrap.dedent(
                """
                <div class="pane-box">
                <div class="pane-header">
                <span class="pane-label">Inspector / Kebaruan & Telemetri</span>
                <span class="pane-meta">HSV + DUAL-ROI TAX</span>
                </div>
                </div>
                """
            ).strip(),
            unsafe_allow_html=True,
        )

        if not detections:
            st.warning("Tidak ada pelat nomor yang terdeteksi pada gambar ini.")
        else:
            if len(detections) > 1:
                if st.session_state.get("plate_inspector_idx", 0) >= len(detections):
                    st.session_state["plate_inspector_idx"] = 0
                sel_idx = st.selectbox(
                    "Pilih Pelat untuk Diinspeksi",
                    range(len(detections)),
                    key="plate_inspector_idx",
                    format_func=lambda idx: f"Pelat #{idx + 1:02d} — {detections[idx]['plate_text']} ({detections[idx]['plate_color']})",
                )
            else:
                sel_idx = 0

            det = detections[sel_idx]
            p_color = det.get("plate_color", "Hitam")
            theme_cls = f"plate-theme-{p_color}" if p_color in ("Putih", "Hitam", "Kuning", "Merah") else "plate-theme-Hitam"
            tax_disp = det["tax_text"] if det["tax_text"] != "TIDAK TERBACA" else "--.--"

            st.markdown(
                textwrap.dedent(
                    f"""
                    <div class="plate-replica {theme_cls}">
                    <div class="plate-main-txt">{det['plate_text']}</div>
                    <div class="plate-tax-txt">{tax_disp}</div>
                    </div>
                    """
                ).strip(),
                unsafe_allow_html=True,
            )

            p_badge = (
                f'<span class="badge badge-ok">VALID ({det["plate_conf"]:.0%})</span>'
                if det["plate_valid"]
                else '<span class="badge badge-err">TIDAK VALID</span>'
            )

            if det["is_tax_active"] is True:
                t_badge = f'<span class="badge badge-ok">PAJAK AKTIF ({det["tax_conf"]:.0%})</span>'
            elif det["is_tax_active"] is False:
                t_badge = f'<span class="badge badge-err">PAJAK MATI ({det["tax_conf"]:.0%})</span>'
            else:
                t_badge = '<span class="badge badge-warn">TIDAK TERBACA</span>'

            gt_row = (
                f'<tr><td class="spec-key">Ground Truth</td><td class="spec-val spec-mono">{gt_val}</td></tr>'
                if gt_val
                else ""
            )

            spec_html = (
                '<div class="pane-box">'
                '<table class="spec-table">'
                f'<tr><td class="spec-key">Nomor Utama</td><td class="spec-val spec-mono">{det["plate_text"]} &nbsp;{p_badge}</td></tr>'
                f'<tr><td class="spec-key">Warna &amp; Jenis</td><td class="spec-val">{p_color} &mdash; {det["vehicle_type"]}</td></tr>'
                f'<tr><td class="spec-key">Masa Pajak</td><td class="spec-val spec-mono">{det["tax_text"]} &nbsp;{t_badge}</td></tr>'
                f'<tr><td class="spec-key">Koreksi Kemiringan</td><td class="spec-val">{det["deskew_status"]}</td></tr>'
                f'<tr><td class="spec-key">Pangkas Pinggiran Plat</td><td class="spec-val">{det.get("trim_status", "Tidak dipangkas").replace(" (huruf tidak terdeteksi)", "")}</td></tr>'
                f'<tr><td class="spec-key">Standar Polaritas</td><td class="spec-val">{det["polarity_status"]}</td></tr>'
                f'<tr><td class="spec-key">Rasio Aspek &amp; Keutuhan</td><td class="spec-val spec-mono">{det["aspect_ratio"]:.2f}:1 &middot; {det["status"]}</td></tr>'
                f'<tr><td class="spec-key">Sumber OCR Pemenang</td><td class="spec-val">{det["ocr_source"]}</td></tr>'
                f'{gt_row}'
                '</table>'
                '</div>'
            )
            st.markdown(spec_html, unsafe_allow_html=True)

            if det["all_candidates"]:
                st.markdown('<span class="pane-label">Komparasi Kandidat OCR (Voting)</span>', unsafe_allow_html=True)
                df_ocr = pd.DataFrame(
                    [
                        {
                            "Kandidat Input": c["source"],
                            "Hasil Bacaan": c["fmt"],
                            "Confidence": f"{c['conf']:.0%}",
                            "Valid Regex": "Ya" if c["valid"] else "Tidak",
                        }
                        for c in det["all_candidates"]
                    ]
                )
                st.dataframe(df_ocr, use_container_width=True, hide_index=True)

    # ==========================================================================
    # FULL-WIDTH 10-STAGE PCD EVALUATION GRID
    # ==========================================================================
    if show_pcd_grid and detections:
        det = detections[sel_idx]
        pre = det["pre"]
        bn = det["bin"]

        st.markdown(
            f"""
        <div class="pane-box" style="margin-top: 0.5rem;">
            <div class="pane-header">
                <span class="pane-label">Box Evaluasi 10 Tahap Pengolahan Citra Digital (Pelat #{sel_idx + 1:02d})</span>
                <span class="pane-meta">{det['hsv_summary']}</span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        r1 = st.columns(5)
        with r1[0]:
            st.markdown('<span class="stage-index">[01]</span> **Crop ROI YOLO**', unsafe_allow_html=True)
            st.image(cv2.cvtColor(det["crop"], cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"Conf={det['confidence']:.2f} | Rasio={det['aspect_ratio']:.2f}")
        with r1[1]:
            st.markdown('<span class="stage-index">[02]</span> **Deskewing &amp; Trim**', unsafe_allow_html=True)
            st.image(cv2.cvtColor(det["deskewed"], cv2.COLOR_BGR2RGB), use_container_width=True)
            trim_txt = det.get("trim_status", "").replace(" (huruf tidak terdeteksi)", "")
            trim_caption = f" &bull; {trim_txt}" if trim_txt else ""
            st.caption(f"{det['deskew_status']}{trim_caption}")
        with r1[2]:
            st.markdown('<span class="stage-index">[03]</span> **Polaritas & HSV**', unsafe_allow_html=True)
            st.image(cv2.cvtColor(det["polarity_std"], cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"{det['plate_color']} | {det['polarity_status']}")
        with r1[3]:
            st.markdown('<span class="stage-index">[04]</span> **Gray & Bilateral**', unsafe_allow_html=True)
            st.image(pre["smoothed"], clamp=True, use_container_width=True)
            st.caption(f"Ukuran={pre['smoothed'].shape[1]}x{pre['smoothed'].shape[0]}")
        with r1[4]:
            st.markdown('<span class="stage-index">[05]</span> **CLAHE Conditional**', unsafe_allow_html=True)
            st.image(pre["enhanced"], clamp=True, use_container_width=True)
            st.caption(pre["clahe_status"])

        r2 = st.columns(5)
        with r2[0]:
            st.markdown('<span class="stage-index">[06]</span> **Unsharp Masking**', unsafe_allow_html=True)
            st.image(pre["sharpened"], clamp=True, use_container_width=True)
            st.caption(f"Laplacian={pre['sharpness_score']}")
        with r2[1]:
            st.markdown('<span class="stage-index">[07]</span> **Segmentasi Biner**', unsafe_allow_html=True)
            st.image(bn["binary_raw"], clamp=True, use_container_width=True)
            st.caption("Otsu (Latar 0 / Teks 255)")
        with r2[2]:
            st.markdown('<span class="stage-index">[08]</span> **Morfologi Closing**', unsafe_allow_html=True)
            st.image(bn["morphed"], clamp=True, use_container_width=True)
            st.caption("Penyambungan celah karakter")
        with r2[3]:
            st.markdown('<span class="stage-index">[09a]</span> **ROI Nomor Utama**', unsafe_allow_html=True)
            st.image(bn["roi_cleaned"], clamp=True, use_container_width=True)
            st.caption(f"Noise dibuang={bn['removed_blobs']} | White={bn['white_ratio']:.1%}")
        with r2[4]:
            st.markdown('<span class="stage-index">[09b]</span> **ROI Pajak (MM.YY)**', unsafe_allow_html=True)
            st.image(bn["tax_roi_binary"], clamp=True, use_container_width=True)
            st.caption(f"Pajak: {det['tax_text']} ({det['tax_status']})")
