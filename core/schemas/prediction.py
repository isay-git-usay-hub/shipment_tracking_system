"""
Pydantic schemas for prediction data
"""
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class PredictionBase(BaseModel):
    """Base prediction schema"""
    shipment_id: int = Field(..., description="Associated shipment ID")
    delay_probability: float = Field(..., ge=0.0, le=1.0, description="Delay probability (0-1)")
    predicted_delay_hours: float = Field(..., ge=0.0, description="Predicted delay in hours")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Model confidence (0-1)")
    model_name: str = Field(..., description="ML model name")
    model_version: str = Field(..., description="Model version")


class PredictionCreate(PredictionBase):
    """Schema for creating a new prediction"""
    features: Optional[Dict[str, Any]] = Field(None, description="Features used for prediction")
    weather_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    traffic_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    operational_risk: float = Field(default=0.0, ge=0.0, le=1.0)


class PredictionResponse(PredictionBase):
    """Schema for prediction response"""
    id: int
    prediction_timestamp: datetime
    features: Optional[Dict[str, Any]] = None
    weather_risk: float
    traffic_risk: float
    operational_risk: float
    risk_level: str = Field(..., description="Risk level (MINIMAL/LOW/MEDIUM/HIGH)")
    prediction_summary: Dict[str, Any] = Field(..., description="Prediction summary")
    created_at: datetime
    updated_at: datetime
    is_active: bool
    
    # Validation data (if available)
    actual_delay_occurred: Optional[bool] = None
    actual_delay_hours: Optional[float] = None
    prediction_accuracy: Optional[float] = None
    
    model_config = ConfigDict(from_attributes=True)


class PredictionSummary(BaseModel):
    """Summary schema for prediction lists"""
    id: int
    shipment_id: int
    delay_probability: float
    predicted_delay_hours: float
    risk_level: str
    confidence_score: float
    prediction_timestamp: datetime
    model_name: str
    
    model_config = ConfigDict(from_attributes=True)


class PredictionRequest(BaseModel):
    """Schema for requesting a prediction"""
    asset_id: str = Field(..., description="Asset identifier")
    latitude: float = Field(..., description="Current latitude")
    longitude: float = Field(..., description="Current longitude")
    inventory_level: int = Field(..., description="Current inventory level")
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    traffic_status: Optional[str] = None
    waiting_time: int = Field(..., description="Current waiting time in minutes")
    user_transaction_amount: Optional[float] = None
    user_purchase_frequency: Optional[int] = None
    asset_utilization: Optional[float] = None
    demand_forecast: Optional[float] = None


class PredictionValidation(BaseModel):
    """Schema for validating predictions with actual outcomes"""
    prediction_id: int
    actual_delay_occurred: bool
    actual_delay_hours: float = Field(..., ge=0.0)


class ModelPerformance(BaseModel):
    """Schema for model performance metrics"""
    model_name: str
    model_version: str
    total_predictions: int
    accuracy_score: float
    precision_score: float
    recall_score: float
    f1_score: float
    mean_absolute_error: float
    avg_confidence: float
    evaluation_period: str
    last_updated: datetime
