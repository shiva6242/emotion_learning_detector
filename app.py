import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

# ---------------- PAGE SETUP ----------------
st.set_page_config(
    page_title="Emotion Based Learning",
    page_icon="🎓",
    layout="centered"
)

st.markdown(
    """
    <h1 style='text-align: center;'>🎓 Emotion-Based Learning Difficulty Detector</h1>
    <p style='text-align: center; color: gray;'>
    Real-time facial emotion detection using a trained CNN model
    </p>
    """,
    unsafe_allow_html=True
)

st.info(
    "👉 Click **Start Camera**, look straight at the camera, and keep your face well-lit."
)

# ---------------- LOAD MODEL ----------------
model = load_model("emotion_cnn_model.h5")

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

learning_state_map = {
    "Angry": "🔴 High Difficulty",
    "Fear": "🔴 High Difficulty",
    "Sad": "🟠 Medium Difficulty",
    "Disgust": "🟠 Medium Difficulty",
    "Neutral": "🟢 Normal Learning",
    "Happy": "🟢 Easy Learning",
    "Surprise": "🟡 Normal Learning"
}

# ---------------- FACE CASCADE ----------------
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# ---------------- SESSION STATE ----------------
if "run" not in st.session_state:
    st.session_state.run = False

# ---------------- BUTTONS ----------------
col1, col2 = st.columns(2)

with col1:
    if st.button("▶ Start Camera"):
        st.session_state.run = True

with col2:
    if st.button("⏹ Stop Camera"):
        st.session_state.run = False

frame_window = st.image([])
emotion_box = st.empty()
confidence_box = st.empty()
learning_box = st.empty()

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

# ---------------- LIVE LOOP ----------------
while st.session_state.run:
    ret, frame = cap.read()
    if not ret :
        st.error("Camera not accessible")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = face.reshape(1, 48, 48, 1)

        pred = model.predict(face, verbose=0)[0]
        idx = np.argmax(pred)

        emotion = emotion_labels[idx]
        confidence = float(pred[idx]) * 100
        learning_state = learning_state_map[emotion]

        # UI TEXT
        emotion_box.success(f"😊 Emotion Detected: **{emotion}**")
        confidence_box.info(f"📊 Confidence: **{confidence:.2f}%**")
        learning_box.warning(f"📚 Learning Difficulty: **{learning_state}**")

        # DRAW ON FRAME
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{emotion} ({confidence:.1f}%)",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_window.image(frame)

    time.sleep(0.03)

cap.release()

# ---------------- FOOTER ----------------
st.markdown(
    "<hr><p style='text-align: center; color: gray;'>"
    "Project Demo | Emotion-Based Learning System</p>",
    unsafe_allow_html=True
)
