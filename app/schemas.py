"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel
from typing import Dict


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str


class PredictionResponse(BaseModel):
    """Prediction response schema."""
    predicted_class: int
    label: str
    confidence: float
    probabilities: Dict[str, float]
