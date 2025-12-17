from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="Heart Disease Prediction API")

# Enable Prometheus Monitoring
Instrumentator().instrument(app).expose(app)

# Load Model
model = joblib.load("models/model.pkl")

class PatientData(BaseModel):
    features: list[float]  # Expecting list of 13 features

@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is running"}

@app.post("/predict")
def predict(data: PatientData):
    features = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]
    
    return {
        "prediction": int(prediction[0]),
        "probability": float(probability),
        "risk": "High" if prediction[0] == 1 else "Low"
    }
