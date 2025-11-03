from fastapi import APIRouter
import joblib
import numpy as np
from utils.preprocessing import preprocess_soil_data

router = APIRouter()
model = joblib.load('app/models/model_soil.pkl')

@router.post("/")
def predict_soil(data: dict):
    """Predict soil quality"""
    features = preprocess_soil_data(data)
    prediction = model.predict(np.array(features).reshape(1, -1))[0]

    return {"soil_quality_score": round(float(prediction), 3)}
