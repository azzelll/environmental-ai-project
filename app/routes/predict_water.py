from fastapi import APIRouter
import joblib
import numpy as np
from utils.preprocessing import preprocess_water_data

router = APIRouter()
model = joblib.load('app/models/model_water.pkl')

@router.post("/")
def predict_water(data: dict):
    """Predict water quality"""
    features = preprocess_water_data(data)
    prediction = model.predict(np.array(features).reshape(1, -1))[0]

    return {"water_quality_score": round(float(prediction), 3)}
