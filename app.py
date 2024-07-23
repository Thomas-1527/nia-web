import streamlit as st
import requests

# URL de l'API FastAPI
api_url = "http://127.0.0.1:8000/"

st.title("Détection des Yeux et Prédiction de Fatigue")
st.subheader("Flux en Direct de la Webcam ou Télécharger une Vidéo")

# Fonction pour faire une requête à l'API
def get_api_message():
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            return response.json()
        else:
            return {"message": "Erreur lors de la requête à l'API"}
    except Exception as e:
        return {"message": f"Exception lors de la requête à l'API: {e}"}

# Afficher le message de l'API
api_response = get_api_message()
st.write(api_response["message"])

# Vous pouvez ajouter plus de logique ici pour traiter les vidéos et afficher les prédictions
