import streamlit as st
import cv2
import numpy as np
import time
from ultralytics import YOLO
import io
from PIL import Image

# ------------------------------
# App UI Theme
# ------------------------------
st.set_page_config(
    page_title="AI Object Detection Suite",
    layout="wide",
    page_icon="🤖"
)

st.markdown("""
    <h1 style="text-align:center; color:#00e6b8; font-size:38px;">
        🚀 AI Object Detection Suite (YOLOv8 + Streamlit)
    </h1>
""", unsafe_allow_html=True)

# ------------------------------
# Sidebar Tabs (NO Webcam)
# ------------------------------
tabs = st.sidebar.radio(
    "Choose Mode",
    ["🖼 Image Detection", "🎞 Video Detection"]
)

# ------------------------------
# Load YOLOv8 model (CACHED)
# ------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ------------------------------
# Common Settings
# ------------------------------
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
show_fps = st.sidebar.checkbox("Show FPS", True)

# ------------------------------
# IMAGE UPLOAD DETECTION
# ------------------------------
if tabs == "🖼 Image Detection":

    st.markdown("## 🖼 Upload Image for Detection")
    uploaded = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded is not None:
        image_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        start = time.time()
        results = model(img, conf=conf_threshold)
        annotated = results[0].plot()

        if show_fps:
            fps = int(1 / (time.time() - start + 1e-6))
            cv2.putText(
                annotated,
                f"FPS: {fps}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        st.image(annotated, channels="BGR", use_column_width=True)

        # Safe download (no disk write)
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")

        st.download_button(
            label="⬇️ Download Result",
            data=buf.getvalue(),
            file_name="detected_image.jpg",
            mime="image/jpeg"
        )

# ------------------------------
# VIDEO UPLOAD DETECTION
# ------------------------------
elif tabs == "🎞 Video Detection":

    st.markdown("## 🎞 Upload Video for Detection")
    video_file = st.file_uploader(
        "Upload MP4 / AVI / MOV",
        type=["mp4", "avi", "mov"]
    )

    if video_file is not None:
        temp_path = "uploaded_video.mp4"
        with open(temp_path, "wb") as f:
            f.write(video_file.read())

        cap = cv2.VideoCapture(temp_path)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            start = time.time()
            results = model(frame, conf=conf_threshold)
            annotated = results[0].plot()

            if show_fps:
                fps = int(1 / (time.time() - start + 1e-6))
                cv2.putText(
                    annotated,
                    f"FPS: {fps}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    2
                )

            stframe.image(annotated, channels="BGR")

        cap.release()
        st.success("🎉 Video Processing Complete!")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption("Made by Ankit ❤️ — Streamlit + YOLOv8")
