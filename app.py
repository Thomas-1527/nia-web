import os
import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import tempfile

# Fonction pour envoyer la vidéo à l'API et récupérer les prédictions
def send_video_to_api(file_path):
    url = "http://127.0.0.1:8000/upload"  # URL de l'API FastAPI
    with open(file_path, 'rb') as file:
        files = {'file': (file_path, file, 'video/mp4')}
        response = requests.post(url, files=files)
    return response

# Fonction pour traiter une vidéo uploadée
def process_uploaded_video(uploaded_video):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_video.read())
        tmp_path = tmp_file.name

    # Envoyer la vidéo à l'API et récupérer les résultats
    response = send_video_to_api(tmp_path)

    if response.status_code == 200:
        st.success("Video processed successfully")
        stframe = st.empty()
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                try:
                    img = Image.open(BytesIO(chunk))
                    stframe.image(img, channels="RGB")
                except Exception as e:
                    print(f"Error loading image: {e}")
    else:
        st.error("Error processing video")

    os.remove(tmp_path)

# Streamlit UI
st.title("Détection des Yeux et Prédiction de Fatigue")
st.subheader("Flux en Direct de la Webcam ou Télécharger une Vidéo")

# Téléchargement de vidéo pour traitement
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
if uploaded_video is not None:
    process_uploaded_video(uploaded_video)
