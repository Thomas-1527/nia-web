import streamlit as st
import requests
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import tempfile
import cv2
import numpy as np

# URL de votre API FastAPI
API_URL = "http://127.0.0.1:8000/predict"

# Classe pour la transformation vidéo en direct
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
        self.left_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_lefteye_2splits.xml")
        self.right_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_righteye_2splits.xml")

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        height, _, _ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (150, 150, 150), 1)

        right_eye_pred = self.predict_eye_state(img, gray, self.right_eye_cascade, 'RIGHT')
        left_eye_pred = self.predict_eye_state(img, gray, self.left_eye_cascade, 'LEFT')

        if right_eye_pred == 'Closed' and left_eye_pred == 'Closed':
            cv2.putText(img, "BOTH CLOSE", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(img, "BOTH OPEN", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return img

    def predict_eye_state(self, frame, gray_frame, eye_cascade, eye_side):
        eyes = eye_cascade.detectMultiScale(gray_frame)
        for (ex, ey, ew, eh) in eyes:
            eye = frame[ey:ey + eh, ex:ex + ew]
            eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
            eye = cv2.resize(eye, (100, 100))
            eye = eye / 255.0
            eye = eye.reshape(100, 100, 1)
            eye = np.expand_dims(eye, axis=0)

            _, buffer = cv2.imencode('.jpg', eye)
            response = requests.post(API_URL, files={"file": buffer.tobytes()})

            if response.status_code == 200:
                pred = response.json()
                eye_state = pred['eye_state']
                if eye_state == "open":
                    return "Open"
                else:
                    return "Closed"
        return "Unknown"

# Fonction pour traiter une vidéo uploadée
def process_uploaded_video(uploaded_video):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_video.read())
        tmp_path = tmp_file.name

    cap = cv2.VideoCapture(tmp_path)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, _, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (150, 150, 150), 1)

        right_eye_pred = predict_eye_state(frame, gray, right_eye_cascade, 'RIGHT')
        left_eye_pred = predict_eye_state(frame, gray, left_eye_cascade, 'LEFT')

        if right_eye_pred == 'Closed' and left_eye_pred == 'Closed':
            cv2.putText(frame, "BOTH CLOSE", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "BOTH OPEN", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        stframe.image(frame, channels="BGR")

    cap.release()

def predict_eye_state(frame, gray_frame, eye_cascade, eye_side):
    eyes = eye_cascade.detectMultiScale(gray_frame)
    for (ex, ey, ew, eh) in eyes:
        eye = frame[ey:ey + eh, ex:ex + ew]
        eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
        eye = cv2.resize(eye, (100, 100))
        eye = eye / 255.0
        eye = eye.reshape(100, 100, 1)
        eye = np.expand_dims(eye, axis=0)

        _, buffer = cv2.imencode('.jpg', eye)
        response = requests.post(API_URL, files={"file": buffer.tobytes()})

        if response.status_code == 200:
            pred = response.json()
            eye_state = pred['eye_state']
            if eye_state == "open":
                return "Open"
            else:
                return "Closed"
    return "Unknown"

# Streamlit UI
st.title("Détection des Yeux et Prédiction de Fatigue")
st.subheader("Flux en Direct de la Webcam ou Télécharger une Vidéo")

# Démarrer le streamer WebRTC pour webcam
webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

# Téléchargement de vidéo pour traitement
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
if uploaded_video is not None:
    process_uploaded_video(uploaded_video)
