# Real-Time Object Detection App (YOLOv8 + SSD + Streamlit + WebRTC)

- A fully interactive Real-Time Object Detection System built using Python, YOLOv8, SSD MobileNet, OpenCV, and Streamlit with WebRTC live camera streaming.
This application allows users to perform object detection on live webcam, images, and videos with smooth UI control.

# 🔗 Live App Link

https://object-detection-app-u8cxuvmxtqwzrt5tyvobxk.streamlit.app/



# 📝 Project Description

- This project demonstrates how to build a production-ready real-time object detection web app using:
- YOLOv8 for high-accuracy detection
- SSD MobileNet V3 for fast CPU inference
- OpenCV for image/video frame processing
- Streamlit for interactive UI
- WebRTC (streamlit-webrtc) for real-time camera streaming

# Custom settings like:

- Confidence threshold slider
- Show/Hide FPS
- Camera selection
- Custom ICE servers for WebRTC
- It is lightweight, fast, and ideal for deployments on Render, Streamlit Cloud, or local servers.

# 📂 Project Structure
📦 object_detection_app/
│
├── .streamlit/
│   ├── config.toml
│   ├── packages.txt
│   └── runtime.txt
│
├── venv/                          # Virtual environment
├── camera.py                      # Camera utilities
├── coco.names                     # Class names for models
├── frozen_inference_graph.pb      # SSD MobileNet model file
├── ssd_mobilenet_v3_large_coco.pbtxt
├── yolov8n.pt                     # YOLO model file
│
├── output_image.jpg               # Sample output
├── main.py                        # Main Streamlit app (WebRTC + YOLO + SSD)
├── requirements.txt               # Dependency list
└── README.md                      # Documentation


# ✔ This README matches the exact structure shown in your screenshot.

# 🚀 Features
✅ Real-time Object Detection (Live Webcam using WebRTC)
✅ YOLOv8 + SSD MobileNet Support
✅ Image Upload Detection
✅ Video Upload Detection
✅ Adjustable Detection Threshold
✅ Show FPS Option
✅ Custom ICE Servers for WebRTC
✅ Gorgeous Streamlit UI
✅ Works on CPU — No GPU Required
✅ Ready for Deployment

# 🛠️ Technologies Used
- Tool	Usage
- Python 3.10+	Main programming language
- OpenCV	Frame capturing, preprocessing, DNN inference
- streamlit-webrtc	Real-time camera streaming
- YOLOv8 (ultralytics)	High accuracy object detection
- SSD MobileNet V3	Lightweight, fast detection
- Streamlit	Front-end UI
  
# 📥 Installation
1️⃣ Clone the repo
git clone https://github.com/your-username/object_detection_app.git
cd object_detection_app

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the app
streamlit run main.py

# 🧪 How It Works
🔄 YOLOv8 Mode

- The YOLO model is loaded via ultralytics.
- Each frame is converted to tensor → inference → boxes + labels → drawn.

# ⚡ SSD MobileNet Mode

- OpenCV DNN loads .pb + .pbtxt

- Pass frame → get detections → draw bounding boxes.

# 🎥 WebRTC Live Camera

- Uses webrtc_streamer()
- Custom ICE servers for stability
- VideoTransformer processes frames in real-time

- Inside your code:

webrtc_streamer(
    key=f"cam-{cam_choice}",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={...},
    video_transformer_factory=LiveTransformer,
)

# 📸 Screenshots (Optional)

![IMG-20251208-WA0001](https://github.com/user-attachments/assets/302cf7f5-1c2f-41bb-ad4f-e7cf30793a11)


# 📤 Deployment (Streamlit Cloud / Render)

This app includes:

✔ .streamlit/config.toml
✔ runtime.txt
✔ packages.txt

So  the deployment is easy
