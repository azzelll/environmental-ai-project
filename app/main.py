# main.py — Environmental Quality Prediction API with Gemini Description
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle, os
from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env file!")

client = genai.Client(api_key=GEMINI_API_KEY)


app = FastAPI(
    title="Environmental Quality Prediction API",
    description="Predict Air, Water, Soil, and Environmental Quality Score (EQS) with Gemini-generated analysis.",
    version="4.0"
)

try:
    BASE_DIR = os.path.dirname(__file__)
    MODEL_DIR = os.path.join(BASE_DIR, "models")

    # Air
    with open(os.path.join(MODEL_DIR, "air_model.pkl"), "rb") as f:
        air_model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "scaler_air.pkl"), "rb") as f:
        air_scaler = pickle.load(f)

    # Water
    with open(os.path.join(MODEL_DIR, "water_model.pkl"), "rb") as f:
        water_model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "scaler_water.pkl"), "rb") as f:
        water_scaler = pickle.load(f)

    # Soil
    with open(os.path.join(MODEL_DIR, "soil_model.pkl"), "rb") as f:
        soil_model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "scaler_soil.pkl"), "rb") as f:
        soil_scaler = pickle.load(f)

except Exception as e:
    raise RuntimeError(f"Failed to load models: {e}")


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


def classify_eqs(eqs: float) -> str:
    if eqs >= 70:
        return "Good"
    elif eqs >= 35:
        return "Moderate"
    else:
        return "Poor"


def generate_description(air, water, soil, eqs, category):
    prompt = f"""
    Analyze the following environmental scores and provide a concise summary (max 5 sentences):
    - Air Quality Score: {air:.2f}
    - Water Quality Score: {water:.2f}
    - Soil Quality Score: {soil:.2f}
    - Overall EQS: {eqs:.2f} ({category})

    Explain:
    1. What these scores indicate about the environment.
    2. How good or bad the overall quality is.
    3. One actionable suggestion for improvement.
    Keep it simple and human-readable, PLAIN TEXT(text only don't add bold or smth), 2-3 sentences, in indonesia, and make it for description of environmental quality.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        ),
    )
    return response.text.strip()


@app.get("/")
def root():
    return {"message": "Environmental Quality Prediction API (Gemini 2.5) is running!"}


@app.post("/predict")
def predict_eqs(request: EQSRequest):
    try:
        # -------- AIR --------
        X_air = preprocess(request.air.dict(), "air")
        X_air_scaled = air_scaler.transform(X_air)
        aqi_pred = air_model.predict(X_air_scaled)[0]
        air_score = 100 - (aqi_pred / 500 * 100)

        # -------- WATER --------
        X_water = preprocess(request.water.dict(), "water")
        X_water_scaled = water_scaler.transform(X_water)
        water_pred = float(water_model.predict(X_water_scaled)[0])
        water_score = (water_pred / 2 * 100)

        # -------- SOIL --------
        X_soil = preprocess(request.soil.dict(), "soil")
        X_soil_scaled = soil_scaler.transform(X_soil)
        soil_score = float(soil_model.predict(X_soil_scaled)[0])

        # -------- EQS --------
        eqs = max(0, min(100, 0.4 * air_score + 0.3 * water_score + 0.3 * soil_score))
        category = classify_eqs(eqs)

        # -------- Gemini description --------
        description = generate_description(air_score, water_score, soil_score, eqs, category)

        # -------- Return --------
        print(air_score, water_score, soil_score, eqs, category, description)
        return {
            "Air_Score": round(aqi_pred, 2),
            "Water_Score": round(water_score, 2),
            "Soil_Score": round(soil_score, 2),
            "EQS": round(eqs, 2),
            "Category": category,
            "Description": description
        }

    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    


# utils/preprocess.py
import pandas as pd

# Air
def preprocess_air_data(data: dict) -> pd.DataFrame:
    return pd.DataFrame([{
        "CO(GT)": float(data.get("CO(GT)", data.get("CO_GT", 0))),
        "NO2(GT)": float(data.get("NO2(GT)", data.get("NO2_GT", 0))),
        "PT08.S5(O3)": float(data.get("PT08.S5(O3)", data.get("PT08_S5_O3", 0))),
        "T": float(data.get("T", 0)),
        "RH": float(data.get("RH", 0)),
        "AH": float(data.get("AH", 0))
    }]).fillna(0)

# Water
def preprocess_water_data(data: dict) -> pd.DataFrame:
    return pd.DataFrame([{
        "Temp": float(data.get("Temp", 0)),
        "Turbidity (cm)": float(data.get("Turbidity (cm)", data.get("Turbidity_cm", 0))),
        "DO(mg/L)": float(data.get("DO(mg/L)", data.get("DO_mg_L", 0))),
        "BOD (mg/L)": float(data.get("BOD (mg/L)", data.get("BOD_mg_L", 0))),
        "CO2": float(data.get("CO2", 0)),
        "pH": float(data.get("pH", 0)),
        "Alkalinity (mg L-1 )": float(data.get("Alkalinity (mg L-1 )", data.get("Alkalinity_mg_L", 0))),
        "Hardness (mg L-1 )": float(data.get("Hardness (mg L-1 )", data.get("Hardness_mg_L", 0))),
        "Calcium (mg L-1 )": float(data.get("Calcium (mg L-1 )", data.get("Calcium_mg_L", 0))),
        "Ammonia (mg L-1 )": float(data.get("Ammonia (mg L-1 )", data.get("Ammonia_mg_L", 0))),
        "Nitrite (mg L-1 )": float(data.get("Nitrite (mg L-1 )", data.get("Nitrite_mg_L", 0))),
        "Phosphorus (mg L-1 )": float(data.get("Phosphorus (mg L-1 )", data.get("Phosphorus_mg_L", 0))),
        "H2S (mg L-1 )": float(data.get("H2S (mg L-1 )", data.get("H2S_mg_L", 0))),
        "Plankton (No. L-1)": float(data.get("Plankton (No. L-1)", data.get("Plankton_No_L", 0)))
    }]).fillna(0)

# Soil
def preprocess_soil_data(data: dict) -> pd.DataFrame:
    return pd.DataFrame([{
        "N": float(data.get("N", 0)),
        "P": float(data.get("P", 0)),
        "K": float(data.get("K", 0)),
        "ph": float(data.get("ph", data.get("pH", 0))),
        "EC": float(data.get("EC", 0)),
        "S": float(data.get("S", 0)),
        "Cu": float(data.get("Cu", 0)),
        "Fe": float(data.get("Fe", 0)),
        "Mn": float(data.get("Mn", 0)),
        "Zn": float(data.get("Zn", 0)),
        "B": float(data.get("B", 0))
    }]).fillna(0)

# Unified selector
def preprocess(data: dict, data_type: str):
    data_type = data_type.lower()
    if data_type == "air":
        return preprocess_air_data(data)
    elif data_type == "water":
        return preprocess_water_data(data)
    elif data_type == "soil":
        return preprocess_soil_data(data)
    else:
        raise ValueError(f"Unknown data_type '{data_type}'. Use 'air', 'water', or 'soil'.")