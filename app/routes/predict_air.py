import pickle
import os
from app.utils.preprocess import preprocess
from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY not found in .env file!")

client = genai.Client(api_key=GEMINI_API_KEY)

# path absolut ke model folder
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # -> app/
MODEL_PATH = os.path.join(BASE_DIR, "models", "air_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler_air.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

sample = {
    "CO_GT": 0.5,
    "NO2_GT": 28.0,
    "PT08_S5_O3": 370.0,
    "T": 12.5,
    "RH": 66.8,
    "AH": 0.962
}
def classify_eqs(eqs: float) -> str:
    if eqs >= 80:
        return "Excellent"
    elif eqs >= 60:
        return "Good"
    elif eqs >= 40:
        return "Moderate"
    elif eqs >= 20:
        return "Poor"
    else:
        return "Critical"


def generate_description(air, category):
    prompt = f"""
    Analyze the following environmental scores and provide a concise summary (max 5 sentences):
    - Air Quality Score: {air:.2f}

    Explain:
    1. What these scores indicate about the environment.
    2. How good or bad the overall quality is.
    3. One actionable suggestion for improvement.
    Keep it simple and human-readable.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        ),
    )
    return response.text.strip()

X = preprocess(sample, "air")
X_scaled = scaler.transform(X)
prediction = model.predict(X_scaled)[0]
air_score = 100 - (prediction / 500 * 100)

print("Predicted AQI:", prediction)
print("Gemini Analysis:", generate_description(air_score, classify_eqs(air_score)))