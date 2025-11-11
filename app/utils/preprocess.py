# utils/preprocess.py
import pandas as pd

# 🌫️ Air
def preprocess_air_data(data: dict) -> pd.DataFrame:
    return pd.DataFrame([{
        "CO(GT)": float(data.get("CO(GT)", data.get("CO_GT", 0))),
        "NO2(GT)": float(data.get("NO2(GT)", data.get("NO2_GT", 0))),
        "PT08.S5(O3)": float(data.get("PT08.S5(O3)", data.get("PT08_S5_O3", 0))),
        "T": float(data.get("T", 0)),
        "RH": float(data.get("RH", 0)),
        "AH": float(data.get("AH", 0))
    }]).fillna(0)

# 💧 Water
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

# 🌱 Soil
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

# 🔁 Unified selector
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