"""
Prediction models for Maersk Shipment AI System
"""
from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .base import BaseModel


class DelayPrediction(BaseModel):
    """Model for storing ML predictions"""
    __tablename__ = "delay_predictions"
    
    # Link to shipment
    shipment_id = Column(Integer, ForeignKey("shipments.id"), nullable=False, index=True)
    
    # Prediction results
    delay_probability = Column(Float, nullable=False)  # 0.0 to 1.0
    predicted_delay_hours = Column(Float, nullable=False)
    confidence_score = Column(Float, nullable=False)  # Model confidence
    
    # Model information
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    prediction_timestamp = Column(DateTime, default=func.now(), nullable=False)
    
    # Features used for prediction (JSON format)
    features = Column(JSON)
    
    # Risk factors
    weather_risk = Column(Float, default=0.0)
    traffic_risk = Column(Float, default=0.0)
    operational_risk = Column(Float, default=0.0)
    
    # Validation data (filled after actual outcome)
    actual_delay_occurred = Column(Boolean)
    actual_delay_hours = Column(Float)
    prediction_accuracy = Column(Float)  # How close the prediction was
    
    # Relationships
    shipment = relationship("Shipment", back_populates="predictions")
    
    @property
    def risk_level(self) -> str:
        """Get risk level based on delay probability"""
        if self.delay_probability >= 0.8:
            return "HIGH"
        elif self.delay_probability >= 0.5:
            return "MEDIUM"
        elif self.delay_probability >= 0.2:
            return "LOW"
        else:
            return "MINIMAL"
    
    @property
    def prediction_summary(self) -> Dict[str, Any]:
        """Get a summary of the prediction"""
        return {
            "delay_probability": round(self.delay_probability * 100, 1),
            "predicted_delay_hours": round(self.predicted_delay_hours, 2),
            "risk_level": self.risk_level,
            "confidence": round(self.confidence_score * 100, 1),
            "model": f"{self.model_name} v{self.model_version}"
        }
