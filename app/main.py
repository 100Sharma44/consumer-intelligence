"""Minimal FastAPI application for purchase intent and consumer intelligence."""

from typing import Optional

from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel

from app.services import consumer_intelligence, purchase_intent


app = FastAPI(title="Consumer Intelligence API", version="0.1.0")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class PredictionResponse(BaseModel):
    user_id: int
    purchase_intent_probability: float
    predicted_purchase_intent: int
    intent_level: str
    primary_interest: Optional[str]
    interest_score: float
    price_sensitivity_score: float
    price_sensitivity_level: str
    consumer_segment: str
    recommended_action: str
    recommended_category: Optional[str]
    messaging_strategy: str
    decision_reason: str


@app.get("/health", response_model=HealthResponse)
def health():
    """Confirm the application can load its saved model artifact."""
    purchase_intent.load_model_artifact()
    return {"status": "ok", "model_loaded": True}


@app.get("/predict/{user_id}", response_model=PredictionResponse)
def predict(user_id: int = Path(..., ge=1)):
    """Return ML purchase propensity and behavioral consumer intelligence for one user."""
    try:
        intelligence = consumer_intelligence.get_consumer_intelligence(user_id)
        decision = consumer_intelligence.make_advertising_decision(intelligence)
        return {**intelligence, **decision}
    except purchase_intent.UserNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
