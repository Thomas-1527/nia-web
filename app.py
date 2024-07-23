import os
import cv2
import streamlit as st
import numpy as np
import requests
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import tempfile
from PIL import Image
import base64

# Définir l'URL de l'API
API_URL = "https://nia-i5tlpsxovq-ew.a.run.app/predict"

# Classe pour la transformation vidéo en direct
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="rgb24")
        _, img_encoded = cv2.imencode('.jpg', img)
        response = requests.post(API_URL, files={"file": img_encoded.tobytes()})
        result = response.json()

        # Decode the image with rectangles from base64
        img_str = result['image']
        img_bytes = base64.b64decode(img_str)
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        height, _, _ = img.shape
        if result['right_eye_state'] == 'Closed' and result['left_eye_state'] == 'Closed':
            cv2.putText(img, "BOTH CLOSE", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(img, "BOTH OPEN", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


        return img

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

        _, img_encoded = cv2.imencode('.jpg', frame)
        response = requests.post(API_URL, files={"file": img_encoded.tobytes()})
        result = response.json()

        # Decode the image with rectangles from base64
        img_str = result['image']
        img_bytes = base64.b64decode(img_str)
        img_np = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        height, _, _ = frame.shape
        if result['right_eye_state'] == 'Closed' and result['left_eye_state'] == 'Closed':
            cv2.putText(frame, "BOTH CLOSE", (20, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "BOTH OPEN", (20, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        stframe.image(frame, channels="RGB")

    cap.release()

# Streamlit UI
st.title("Détection des Yeux et Prédiction de Fatigue")
st.subheader("Flux en Direct de la Webcam ou Télécharger une Vidéo")

# Démarrer le streamer WebRTC pour webcam
webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

# Téléchargement de vidéo pour traitement
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
if uploaded_video is not None:
    process_uploaded_video(uploaded_video)
