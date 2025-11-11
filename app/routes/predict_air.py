import pickle
from utils.preprocess import preprocess

with open("models/model_air.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/scaler_air.pkl", "rb") as f:
    scaler = pickle.load(f)

sample = {
    "CO_GT": 2.5,
    "NO2_GT": 110,
    "PT08_S5_O3": 1000,
    "T": 14.5,
    "RH": 60.0,
    "AH": 0.75
}

X = preprocess(sample, "air")
X_scaled = scaler.transform(X)
prediction = model.predict(X_scaled)[0]

print("Predicted AQI:", prediction)