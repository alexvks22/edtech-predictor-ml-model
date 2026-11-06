import os
import pickle
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# 1. Pydantic Input Schema (Strict 9 Features matching train.py)
class StudentLead(BaseModel):
    discovery_channel: str = Field(default="LinkedIn", example="LinkedIn")
    primary_interest: str = Field(default="Analytics Engineering", example="Analytics Engineering")
    days_signup_vs_reachout: float = Field(default=-1.5, example=-1.5)
    replied_to_followup_email: int = Field(default=1, ge=0, le=1, example=1)
    sent_second_message: int = Field(default=1, ge=0, le=1, example=1)
    free_courses_enrolled_count: int = Field(default=2, ge=0, le=3, example=2)
    free_semantic_model_completed: int = Field(default=1, ge=0, le=1, example=1)
    platform_logins_count_14d: int = Field(default=5, ge=0, example=5)
    total_video_minutes_watched_14d: float = Field(default=60.0, ge=0.0, example=60.0)

# Output Schema
class PredictionResponse(BaseModel):
    conversion_probability: float
    is_converted: bool

# 2. Load Model Artifact
MODEL_FILE = 'model.bin'
try:
    with open(MODEL_FILE, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    print(f"Loaded '{MODEL_FILE}' successfully.")
except FileNotFoundError:
    raise RuntimeError(f"'{MODEL_FILE}' not found! Run python train.py first.")

# 3. Initialize FastAPI App
app = FastAPI(
    title="EdTech Lead Scoring Engine",
    description="Machine Learning service predicting lead conversion probability via XGBoost.",
    version="1.0.0"
)

# Enable CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory if it exists
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_ui():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "EdTech Lead Scoring API is running. Visit /docs for Swagger UI."}

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
def predict_lead(lead: StudentLead):
    try:
        student_dict = lead.model_dump()
        
        # Channel alias normalization
        channel_map = {'linkedin': 'LinkedIn', 'LI': 'LinkedIn', 'yt': 'YouTube'}
        student_dict['discovery_channel'] = channel_map.get(
            student_dict['discovery_channel'],
            student_dict['discovery_channel']
        )
        
        # DictVectorizer transform
        X = dv.transform([student_dict])
        
        # XGBoost DMatrix inference
        dmatrix = xgb.DMatrix(X, feature_names=list(dv.get_feature_names_out()))
        prob = float(model.predict(dmatrix)[0])
        is_converted = bool(prob >= 0.5)
        
        return PredictionResponse(
            conversion_probability=round(prob, 4),
            is_converted=is_converted
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("predict:app", host="0.0.0.0", port=9696, reload=True)