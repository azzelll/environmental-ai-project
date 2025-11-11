# main.py — Unified Environmental Quality Prediction API (EQS only)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
from utils.preprocess import preprocess

app = FastAPI(
    title="Environmental Quality Prediction API",
    description="Predict overall Environmental Quality Score (EQS) from Air, Water, and Soil data.",
    version="3.0"
)

# =====================================================
# Load all models & scalers
# =====================================================
try:
    # Air
    with open("models/model_air.pkl", "rb") as f:
        air_model = pickle.load(f)
    with open("models/scaler_air.pkl", "rb") as f:
        air_scaler = pickle.load(f)

    # Water
    with open("models/model_water.pkl", "rb") as f:
        water_model = pickle.load(f)
    with open("models/scaler_water.pkl", "rb") as f:
        water_scaler = pickle.load(f)

    # Soil
    with open("models/model_soil.pkl", "rb") as f:
        soil_model = pickle.load(f)
    with open("models/scaler_soil.pkl", "rb") as f:
        soil_scaler = pickle.load(f)

except Exception as e:
    raise RuntimeError(f"Gagal load model: {e}")

class AirData(BaseModel):
    CO_GT: float
    NO2_GT: float
    PT08_S5_O3: float
    T: float
    RH: float
    AH: float

class WaterData(BaseModel):
    Temp: float
    Turbidity_cm: float
    DO_mg_L: float
    BOD_mg_L: float
    CO2: float
    pH: float
    Alkalinity_mg_L: float
    Hardness_mg_L: float
    Calcium_mg_L: float
    Ammonia_mg_L: float
    Nitrite_mg_L: float
    Phosphorus_mg_L: float
    H2S_mg_L: float
    Plankton_No_L: float

class SoilData(BaseModel):
    N: float
    P: float
    K: float
    ph: float
    EC: float
    S: float
    Cu: float
    Fe: float
    Mn: float
    Zn: float
    B: float

class EQSRequest(BaseModel):
    air: AirData
    water: WaterData
    soil: SoilData

# =====================================================
# Root endpoint
# =====================================================
@app.get("/")
def root():
    return {"message": "🌎 Environmental Quality Prediction API is running."}

# =====================================================
# Predict endpoint (returns EQS only)
# =====================================================
@app.post("/predict")
def predict_eqs(request: EQSRequest):
    try:

        X_air = preprocess(request.air.dict(), "air")
        X_air_scaled = air_scaler.transform(X_air)
        aqi_pred = air_model.predict(X_air_scaled)[0]
        air_score = 100 - (aqi_pred / 500 * 100)


        X_water = preprocess(request.water.dict(), "water")
        X_water_scaled = water_scaler.transform(X_water)
        water_score = float(water_model.predict(X_water_scaled)[0])


        X_soil = preprocess(request.soil.dict(), "soil")
        X_soil_scaled = soil_scaler.transform(X_soil)
        soil_score = float(soil_model.predict(X_soil_scaled)[0])

        eqs = 0.4 * air_score + 0.3 * water_score + 0.3 * soil_score

        return {"EQS": round(eqs, 2)}


    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

# =====================================================
# Run with:
# uvicorn main:app --reload
# =====================================================