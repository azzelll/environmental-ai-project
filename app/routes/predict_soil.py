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
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_soil.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler_soil.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# =====================================================
# Fungsi klasifikasi dan deskripsi
# =====================================================
def classify_soil(score: float) -> str:
    """Klasifikasi kualitas tanah (skor 0–100)"""
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

def generate_description(score: float, category: str):
    """Deskripsi kualitas tanah menggunakan Gemini"""
    prompt = f"""
    Analyze the following soil quality score and summarize it in under 5 sentences:
    - Soil Quality Score: {score:.2f}
    - Classification: {category}

    Explain:
    1. What this means for crop productivity and soil health.
    2. Key characteristics of this soil condition.
    3. One actionable recommendation to improve or maintain soil quality.
    Keep it simple, human-readable, and practical.
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
# Sample input user
# =====================================================
sample = {
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

# =====================================================
# Prediksi + Analisis
# =====================================================
X = preprocess(sample, "soil")
X_scaled = scaler.transform(X)
prediction = float(model.predict(X_scaled)[0])


category = classify_soil(prediction)
description = generate_description(prediction, category)

print(f"🌱 Predicted Soil Score: {prediction:.2f}")
print(f"🏷️ Category: {category}")
print(f"🧠 Gemini Analysis: {description}")