"""FastAPI application for diabetes prediction."""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Diabetic Risk Prediction API",
    description="API for predicting diabetic risk",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "best_model.pkl"


class PredictionRequest(BaseModel):
    """Request schema for prediction."""
    age: float
    bmi: float
    blood_pressure: float
    glucose: float
    insulin: float
    skin_thickness: float
    pregnancies: float
    diabetes_pedigree_function: float


class PredictionResponse(BaseModel):
    """Response schema for prediction."""
    prediction: int
    probability: float
    risk_level: str


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "message": "Diabetic Risk Prediction API",
        "version": "0.1.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch_predict"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Make a single prediction.
    
    Args:
        request: Prediction request with feature values
        
    Returns:
        Prediction response with prediction and probability
    """
    try:
        # Load model if not already loaded
        if not hasattr(predict, "model"):
            if MODEL_PATH.exists():
                predict.model = joblib.load(MODEL_PATH)
            else:
                raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        
        # Prepare features
        features = np.array([[
            request.age,
            request.bmi,
            request.blood_pressure,
            request.glucose,
            request.insulin,
            request.skin_thickness,
            request.pregnancies,
            request.diabetes_pedigree_function
        ]])
        
        # Make prediction
        prediction = predict.model.predict(features)[0]
        probability = predict.model.predict_proba(features)[0][1]
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            risk_level=risk_level
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        if MODEL_PATH.exists():
            predict.model = joblib.load(MODEL_PATH)
            logger.info(f"Model loaded from {MODEL_PATH}")
        else:
            logger.warning(f"Model not found at {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
