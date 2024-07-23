import streamlit as st
import requests
import cv2
import tempfile
import numpy as np
from PIL import Image

# URL de l'API
API_URL = "http://127.0.0.1:8000/predict"

st.title("Détection des Yeux et Prédiction de Fatigue")

# Téléchargement de vidéo
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    # Charger la vidéo avec OpenCV
    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir le cadre en image
        is_success, im_buf_arr = cv2.imencode(".jpg", frame)
        byte_im = im_buf_arr.tobytes()

        # Envoyer le cadre à l'API pour prédiction
        response = requests.post(API_URL, files={"file": byte_im})

        if response.status_code == 200:
            data = response.json()
            if data["eye_state"] == "closed":
                cv2.putText(frame, "Eyes Closed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Eyes Open", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Afficher le cadre avec les annotations
        stframe.image(frame, channels="BGR")

    cap.release()
