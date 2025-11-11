import pickle
from utils.preprocess import preprocess

with open("models/model_soil.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/scaler_soil.pkl", "rb") as f:
    scaler = pickle.load(f)

sample = {
    "N": 145,
    "P": 42, "K": 210, "ph": 6.4,
    "EC": 0.55, "S": 0.21, "Cu": 10.2, "Fe": 115.0,
    "Mn": 60.0, "Zn": 50.0, "B": 25.0
}

X = preprocess(sample, "soil")
X_scaled = scaler.transform(X)
prediction = model.predict(X_scaled)[0]
print("Predicted Soil Score:", prediction)