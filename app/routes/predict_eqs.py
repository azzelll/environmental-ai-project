from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI(title="Environmental Quality Score API")

class EQSInput(BaseModel):
    air_url: str
    water_url: str
    soil_url: str
    air_data: dict
    water_data: dict
    soil_data: dict

@app.post('/predict')
def predict_eqs(data: EQSInput):
    air = requests.post(data.air_url, json=data.air_data).json()
    water = requests.post(data.water_url, json=data.water_data).json()
    soil = requests.post(data.soil_url, json=data.soil_data).json()

    eqs = 0.4 * air['Air_Score'] + 0.3 * water['Water_Score'] + 0.3 * soil['Soil_Score']

    return {
        "Air_Score": air['Air_Score'],
        "Water_Score": water['Water_Score'],
        "Soil_Score": soil['Soil_Score'],
        "Environmental_Quality_Score": round(eqs, 2)
    }

# Run: uvicorn predict_eqs_api:app --reload
