import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from tensorflow.keras.models import load_model


app = FastAPI()

# ========== LOAD MODEL ==========
try:
    model = load_model('tomate_model.keras')
    logger.info("✅ Model loaded successfully.")
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")
    sys.exit(1)

# ========== CLASS LABELS ==========
class_labels = {
    0: "Tomato Bacterial Spot",
    1: "Tomato Early Blight",
    2: "Tomato Late Blight",
    3: "Tomato Leaf Mold",
    4: "Tomato Septoria Leaf Spot",
    5: "Tomato Spider Mites (Two Spotted Spider Mite)",
    6: "Tomato Target Spot",
    7: "Tomato Tomato Yellow Leaf Curl Virus",
    8: "Tomato Mosaic Virus",
    9: "Tomato Healthy"
}

# ========== PREDICTION ROUTE ==========
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((224, 224))
        img = np.array(img).astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)

        preds = model.predict(img)[0]
        predicted_class = class_labels[np.argmax(preds)]
        confidence = float(np.max(preds))

        return JSONResponse(content={
            "predicted_class": predicted_class,
            "confidence": f"{confidence * 100:.2f}%"
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return JSONResponse(status_code=500, content={"error": "Prediction failed."})
