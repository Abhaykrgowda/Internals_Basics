from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import json
import os
from datetime import datetime

app = FastAPI()

# Load model
model = joblib.load("models/best_model.pkl")

# Ensure logs folder exists
os.makedirs("logs", exist_ok=True)

class InputData(BaseModel):
    followers_count: int = Field(..., ge=100)
    post_hour: int = Field(..., ge=0, le=23)
    has_media: int = Field(..., ge=0, le=1)
    content_length: int = Field(..., ge=10)

@app.get("/status")
def status():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict(data: InputData):
    try:
        input_list = [[
            data.followers_count,
            data.post_hour,
            data.has_media,
            data.content_length
        ]]

        pred = model.predict(input_list)[0]

        log_entry = {
            "timestamp": str(datetime.now()),
            "input": data.dict(),
            "prediction": float(pred)
        }

        with open("logs/predictions.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        return {"prediction": float(pred)}

    except Exception as e:
        print("ERROR:", e)  # 👈 important for debugging
        return {"error": str(e)}