import numpy as np

# ==== Air data ====
def preprocess_air_data(data: dict):
    """
    Expected JSON keys:
    {"CO(GT)": 2.5, "NO2(GT)": 120, "T": 22.1, "RH": 45.0, "AH": 1.3}
    """
    ordered_features = [
        data.get("CO(GT)", 0),
        data.get("NO2(GT)", 0),
        data.get("T", 0),
        data.get("RH", 0),
        data.get("AH", 0)
    ]
    return np.array(ordered_features)

# ==== Soil data ====
def preprocess_soil_data(data: dict):
    """
    Expected JSON keys:
    {"pH": 6.5, "moisture": 20, "nitrogen": 12, "phosphorus": 10, "potassium": 15}
    """
    ordered_features = [
        data.get("pH", 7),
        data.get("moisture", 0),
        data.get("nitrogen", 0),
        data.get("phosphorus", 0),
        data.get("potassium", 0)
    ]
    return np.array(ordered_features)

# ==== Water data ====
def preprocess_water_data(data: dict):
    """
    Expected JSON keys:
    {"pH": 7.0, "Turbidity": 2.5, "DO": 8.0, "BOD": 3.1, "CO2": 4.5}
    """
    ordered_features = [
        data.get("pH", 7),
        data.get("Turbidity", 0),
        data.get("DO", 0),
        data.get("BOD", 0),
        data.get("CO2", 0)
    ]
    return np.array(ordered_features)
