from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()

# Charger le modèle
model_path = '/home/thomas/code/Thomas-1527/nia-web/CNN_hub.h5'  # Mettez à jour ce chemin si nécessaire
model = load_model(model_path)

@app.get("/")
def read_root():
    return {"message": "API is working!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Prétraiter l'image pour la prédiction
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    eyes = eye_cascade.detectMultiScale(gray)

    eye_state = "unknown"
    for (ex, ey, ew, eh) in eyes:
        eye = img[ey:ey + eh, ex:ex + ew]
        eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
        eye = cv2.resize(eye, (100, 100))
        eye = eye / 255.0
        eye = eye.reshape(100, 100, 1)
        eye = np.expand_dims(eye, axis=0)
        pred = model.predict(eye)
        pred_class = np.where(pred[0][0] > 0.5, 1, 0)

        if pred_class == 1:
            eye_state = "open"
        else:
            eye_state = "closed"
        break

    return JSONResponse(content={"eye_state": eye_state})
