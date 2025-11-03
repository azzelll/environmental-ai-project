from fastapi import FastAPI
from routes import predict_air, predict_soil, predict_water
import joblib
import numpy as np

app = FastAPI(
    title="🌍 Hybrid Environmental Quality API",
    version="1.0.0",
    description="Predict Air, Soil, and Water Quality using AI models with a hybrid meta-model."
)

# Load semua model (sekali saja)
model_air = joblib.load('app/models/model_air.pkl')
model_soil = joblib.load('app/models/model_soil.pkl')
model_water = joblib.load('app/models/model_water.pkl')
meta_model = joblib.load('app/models/meta_model.pkl')

# Register route endpoints
app.include_router(predict_air.router, prefix="/predict/air", tags=["Air"])
app.include_router(predict_soil.router, prefix="/predict/soil", tags=["Soil"])
app.include_router(predict_water.router, prefix="/predict/water", tags=["Water"])

@app.get("/")
def root():
    return {"message": "🌿 Environmental AI Hybrid Model API is running!"}

@app.post("/predict/environment")
def predict_environment(data: dict):
    """
    Hybrid prediction: combine air, soil, and water model outputs to get final score.
    """
    air_score = model_air.predict(np.array(data["air"]).reshape(1, -1))[0]
    soil_score = model_soil.predict(np.array(data["soil"]).reshape(1, -1))[0]
    water_score = model_water.predict(np.array(data["water"]).reshape(1, -1))[0]

    final_score = meta_model.predict([[air_score, soil_score, water_score]])[0]

    return {
        "air_score": round(float(air_score), 3),
        "soil_score": round(float(soil_score), 3),
        "water_score": round(float(water_score), 3),
        "environmental_index": round(float(final_score), 3)
    }
