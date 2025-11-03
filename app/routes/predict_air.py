from fastapi import APIRouter
import joblib
import numpy as np
from utils.preprocessing import preprocess_air_data
from utils.helpers import classify_aqi

router = APIRouter()
model = joblib.load('app/models/model_air.pkl')

@router.post("/")
def predict_air(data: dict):
    """Predict air quality from sensor data"""
    features = preprocess_air_data(data)
    prediction = model.predict(np.array(features).reshape(1, -1))[0]
    category = classify_aqi(prediction)

    return {
        "air_quality_score": round(float(prediction), 3),
        "category": category
    }
