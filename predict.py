import pickle
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 1. Schema Εισόδου (9 χαρακτηριστικά, ακριβώς όπως στο train.py)
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

# Schema Εξόδου
class PredictionResponse(BaseModel):
    conversion_probability: float
    is_converted: bool

# 2. Φόρτωση του εκπαιδευμένου μοντέλου (model.bin)
model_file = 'model.bin'
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

# 3. FastAPI App
app = FastAPI(
    title="EdTech Lead Scoring API",
    description="API πρόβλεψης μετατροπής υποψήφιου μαθητή (Lead Conversion) με XGBoost",
    version="1.0.0"
)

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
def predict_lead(lead: StudentLead):
    try:
        student_dict = lead.model_dump()
        
        # Κανονικοποίηση καναλιού αν δοθεί συντομογραφία
        channel_map = {'linkedin': 'LinkedIn', 'LI': 'LinkedIn', 'yt': 'YouTube'}
        student_dict['discovery_channel'] = channel_map.get(
            student_dict['discovery_channel'], 
            student_dict['discovery_channel']
        )
        
        # Transform με DictVectorizer
        X = dv.transform([student_dict])
        
        # Δημιουργία DMatrix για XGBoost
        dmatrix = xgb.DMatrix(X, feature_names=list(dv.get_feature_names_out()))
        
        # Πρόβλεψη
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