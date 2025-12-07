import streamlit as st
import cv2
import numpy as np
import time
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av

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

tabs = st.sidebar.radio("Choose Mode", ["📷 Live Camera", "🖼 Image Detection", "🎞 Video Detection"])


# ------------------------------
# Load YOLOv8 model
# ------------------------------
model = YOLO("yolov8n.pt")  # smallest fastest model


# ------------------------------
# COMMON SETTINGS
# ------------------------------
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
show_labels = st.sidebar.checkbox("Show Labels", True)
show_conf = st.sidebar.checkbox("Show Confidence %", True)
show_fps = st.sidebar.checkbox("Show FPS", True)


# ------------------------------
# LIVE CAMERA DETECTION
# ------------------------------

if tabs == "📷 Live Camera":

    st.markdown("## 📡 Live Camera Object Detection")
    cam_choice = st.sidebar.radio("Camera:", ["Default", "Front", "Back"])

    def get_constraints(choice):
        if choice == "Front":
            return {"video": {"facingMode": "user"}, "audio": False}
        if choice == "Back":
            return {"video": {"facingMode": "environment"}, "audio": False}
        return {"video": True, "audio": False}

    constraints = get_constraints(cam_choice)

    class LiveTransformer(VideoTransformerBase):

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            start = time.time()

            results = model(img, conf=conf_threshold)
            annotated = results[0].plot()

            if show_fps:
                fps = int(1 / (time.time() - start))
                cv2.putText(
                    annotated,
                    f"FPS: {fps}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2
                )

            return annotated

    # ✅ WebRTC MUST BE OUTSIDE THE CLASS + NO RETURN ABOVE IT
    webrtc_streamer(
        key=f"cam-{cam_choice}",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={
            "iceServers": [
                {"urls": "stun:stun.l.google.com:19302"},
                {
                    "urls": "turn:openrelay.metered.ca:443",
                    "username": "openrelayproject",
                    "credential": "openrelayproject"
                }
            ]
        },
        video_transformer_factory=LiveTransformer,
        media_stream_constraints=constraints,
    )


# ------------------------------
# IMAGE UPLOAD DETECTION
# ------------------------------

elif tabs == "🖼 Image Detection":

    st.markdown("## 🖼 Upload Image for Detection")
    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), 1)

        results = model(img, conf=conf_threshold)
        annotated = results[0].plot()

        st.image(annotated, channels="BGR", use_column_width=True)

        cv2.imwrite("output_image.jpg", annotated)
        with open("output_image.jpg", "rb") as f:
            st.download_button("Download Result", f, file_name="detected.jpg")


# ------------------------------
# VIDEO UPLOAD DETECTION
# ------------------------------

elif tabs == "🎞 Video Detection":

    st.markdown("## 🎞 Upload Video for Detection")
    video = st.file_uploader("Upload MP4/AVI", type=["mp4", "avi", "mov"])

    if video:
        temp_path = "uploaded_video.mp4"
        with open(temp_path, "wb") as f:
            f.write(video.read())

        cap = cv2.VideoCapture(temp_path)
        stframe = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=conf_threshold)
            annotated = results[0].plot()

            stframe.image(annotated, channels="BGR")

        cap.release()
        st.success("🎉 Video Processing Complete!")

st.markdown("---")
st.caption("Made by ankit ❤️ — Streamlit + OpenCV")
