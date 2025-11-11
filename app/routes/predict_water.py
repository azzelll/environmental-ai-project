import pickle
import os
from app.utils.preprocess import preprocess
from dotenv import load_dotenv
from google import genai
from google.genai import types

# =====================================================
# Load environment variable
# =====================================================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY not found in .env file!")

client = genai.Client(api_key=GEMINI_API_KEY)

# =====================================================
# Load model dan scaler
# =====================================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # -> app/
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_water.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler_water.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# =====================================================
# Fungsi klasifikasi dan deskripsi
# =====================================================
def classify_water(score: float) -> str:
    """Klasifikasi kualitas air berdasarkan skor (0-100 atau 0-2)"""
    # Jika model kamu klasifikasi (0,1,2)
    if score in [0, 1, 2]:
        labels = ["Poor", "Moderate", "Good"]
        return labels[int(score)]
    # Kalau model regresi
    if score >= 80:
        return "Excellent"
    elif score >= 60:
        return "Good"
    elif score >= 40:
        return "Moderate"
    elif score >= 20:
        return "Poor"
    else:
        return "Critical"


def generate_description(score, category):
    prompt = f"""
    Analyze the following water quality measurement and give a short summary (max 5 sentences):
    - Water Quality Score: {score:.2f}
    - Classification: {category}

    Explain:
    1. What this score means for the water ecosystem.
    2. How good or bad the water condition is.
    3. One practical step to improve the water quality.
    Keep it short, human-readable, and informative.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        ),
    )
    return response.text.strip()

# =====================================================
# Sample data (contoh input pengguna)
# =====================================================
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

# =====================================================
# Prediksi + analisis
# =====================================================
X = preprocess(sample, "water")
X_scaled = scaler.transform(X)
prediction = model.predict(X_scaled)[0]

# Konversi kalau model klasifikasi (0–2)
if prediction in [0, 1, 2]:
    water_score = (prediction / 2) * 100  # ubah jadi skala 0–100
else:
    water_score = float(prediction)

category = classify_water(prediction)
description = generate_description(water_score, category)

print(f"💧 Predicted Water Score: {water_score:.2f}")
print(f"🏷️ Category: {category}")
print(f"🧠 Gemini Analysis: {description}")