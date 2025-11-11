import pickle
from utils.preprocess import preprocess

with open("models/model_water.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/scaler_water.pkl", "rb") as f:
    scaler = pickle.load(f)

sample = {
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
}

X = preprocess(sample, "water")
X_scaled = scaler.transform(X)
prediction = model.predict(X_scaled)[0]
print("Predicted Water Score:", prediction)