import streamlit as st
from PIL import Image
import numpy as np
import cv2
import json
import time

from backend.scanner import scan_document
from backend.redactor import redact_document

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="SmartShield AI",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ SmartShield AI")
st.write("AI-Powered Privacy Redaction Dashboard")

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header("Settings")

blur_strength = st.sidebar.slider(
    "Blur Intensity",
    min_value=5,
    max_value=99,
    value=51,
    step=2
)

use_scanner = st.sidebar.checkbox(
    "Enable Perspective Correction",
    value=False
)

# ==========================================================
# FILE UPLOADER
# ==========================================================

uploaded_file = st.file_uploader(
    "Upload Document",
    type=["jpg", "jpeg", "png"]
)

# ==========================================================
# PROCESS IMAGE
# ==========================================================

if uploaded_file is not None:

    start_time = time.time()

    image = Image.open(uploaded_file).convert("RGB")

    image_cv = cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2BGR
    )

    with st.spinner("Processing document..."):

        if use_scanner:
            processed = scan_document(image_cv)
        else:
            processed = image_cv

        redacted, detected, faces, signatures = redact_document(
            processed,
            blur_strength=blur_strength
        )

    processing_time = round(time.time() - start_time, 2)

    processed_rgb = cv2.cvtColor(
        processed,
        cv2.COLOR_BGR2RGB
    )

    detected_rgb = cv2.cvtColor(
        detected,
        cv2.COLOR_BGR2RGB
    )

    redacted_rgb = cv2.cvtColor(
        redacted,
        cv2.COLOR_BGR2RGB
    )

    # ======================================================
    # IMAGE DISPLAY
    # ======================================================

    st.subheader("Document Preview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("### Processed")
        st.image(processed_rgb, use_container_width=True)

    with col2:
        st.write("### Detected Regions")
        st.image(detected_rgb, use_container_width=True)

    with col3:
        st.write("### Redacted")
        st.image(redacted_rgb, use_container_width=True)

    st.divider()

    # ======================================================
    # RESULTS
    # ======================================================

    st.subheader("Processing Results")

    st.write(f"**Faces Detected:** {len(faces)}")
    st.write(f"**Signatures Detected:** {len(signatures)}")
    st.write(f"**Processing Time:** {processing_time:.2f} seconds")
    st.write(f"**Blur Strength:** {blur_strength}")

    if use_scanner:
        st.write("**Perspective Correction:** Enabled")
    else:
        st.write("**Perspective Correction:** Disabled")

    st.divider()

    # ======================================================
    # COMPLIANCE SUMMARY
    # ======================================================

    st.subheader("Compliance Summary")

    if len(faces) == 0 and len(signatures) == 0:
        st.success("No sensitive information was detected.")
    else:
        st.success("Sensitive information detected and successfully redacted.")

    st.write(f"- Faces Redacted: {len(faces)}")
    st.write(f"- Signatures Redacted: {len(signatures)}")

    st.divider()

    # ======================================================
    # DOWNLOAD REDACTED IMAGE
    # ======================================================

    success, buffer = cv2.imencode(".png", redacted)

    if success:

        st.download_button(
            label="Download Redacted Image",
            data=buffer.tobytes(),
            file_name="redacted_document.png",
            mime="image/png"
        )

    # ======================================================
    # AUDIT LOG
    # ======================================================

    audit_log = {
        "faces_detected": len(faces),
        "signatures_detected": len(signatures),
        "processing_time_seconds": processing_time,
        "blur_strength": blur_strength,
        "perspective_correction": use_scanner
    }

    st.download_button(
        label="Download Audit Log (JSON)",
        data=json.dumps(audit_log, indent=4),
        file_name="audit_log.json",
        mime="application/json"
    )

    st.success("Document processed successfully.")

# ==========================================================
# FOOTER
# ==========================================================

st.divider()
st.caption("SmartShield AI | Built using Streamlit, OpenCV and YOLO")