# Environmental AI Project  
**Multimodal Environmental Quality Prediction API (Air, Water, Soil)**  
Built with **FastAPI + Machine Learning**, deployed on **Railway**.

---

## Features

### Air Quality Prediction  
Predict AQI using trained ML models (PM2.5, PM10, CO, O3, etc.)

### Water Quality Prediction  
Predict water safety score using chemical indicators (pH, turbidity, etc.)

### Soil Quality Prediction  
Predict soil health using nutrient indicators.

### Scaler + ML Pipeline  
Each domain uses:
- A preprocessing scaler (`scaler_*.pkl`)
- A trained ML model (`*_model.pkl`)
- A clean inference pipeline

### FastAPI REST API  
- Clean routes per domain  
- JSON requests  
- Swagger UI (`/docs`)  
- CORS enabled for frontend support

### Railway Deployment  
- Runs on port `8080`  
- Procfile-based start command  
- Free-tier compatible

---

## Installation (Local Development)

### 1. Clone Project

```bash
git clone https://github.com/<your-username>/environmental-ai-project.git
cd environmental-ai-project
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run FastAPI Server

```bash
uvicorn app.main:app --reload
```

Open docs:

```
http://127.0.0.1:8000/docs
```

---

## 🔥 API Endpoints

### 🌬 EQS Quality — `POST /predict`

**Example JSON Body**
```json
{
  "air": {
    "CO_GT": 0.5,
    "NO2_GT": 28.0,
    "PT08_S5_O3": 370.0,
    "T": 12.5,
    "RH": 66.8,
    "AH": 0.962
  },
  "water": {
    "Temp": 26.4,
    "Turbidity_cm": 45.2,
    "DO_mg_L": 7.8,
    "BOD_mg_L": 2.5,
    "CO2": 10.5,
    "pH": 7.3,
    "Alkalinity_mg_L": 180.0,
    "Hardness_mg_L": 120.0,
    "Calcium_mg_L": 80.0,
    "Ammonia_mg_L": 0.2,
    "Nitrite_mg_L": 0.7,
    "Phosphorus_mg_L": 0.03,
    "H2S_mg_L": 0.02,
    "Plankton_No_L": 1500
  },
  "soil": {
    "N": 45.2,
    "P": 32.8,
    "K": 120.4,
    "ph": 6.5,
    "EC": 0.35,
    "S": 10.8,
    "Cu": 0.8,
    "Fe": 3.2,
    "Mn": 1.5,
    "Zn": 0.9,
    "B": 0.6
  }
}
```


---

## Tech Stack

- Python 3.13 
- FastAPI  
- Uvicorn  
- Railway Deployment  

---

## 👤 Author  
**Made Shandy Krisnanda**  
Computer Science — Universitas Indonesia


