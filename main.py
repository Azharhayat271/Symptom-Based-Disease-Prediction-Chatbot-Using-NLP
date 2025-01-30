import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Load trained model
model = joblib.load("disease_prediction_model.pkl")

# List of symptoms (Make sure this matches the order in your dataset)
symptoms_list = ['fever', 'cough', 'fatigue', 'headache']  # Add all

# API Initialization
app = FastAPI()

# Define API input
class SymptomRequest(BaseModel):
    symptoms: list[str]  # List of user symptoms

@app.post("/predict")
async def predict_disease(request: SymptomRequest):
    # Convert symptoms to model input format
    input_vector = np.zeros(len(symptoms_list))
    for symptom in request.symptoms:
        if symptom in symptoms_list:
            input_vector[symptoms_list.index(symptom)] = 1

    # Make prediction
    prediction = model.predict([input_vector])[0]

    return {"disease": prediction}
