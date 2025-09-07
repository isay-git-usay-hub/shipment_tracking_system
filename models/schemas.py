"""
Pydantic schemas for API request/response validation
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, EmailStr, validator
from enum import Enum


# Enums
class ShipmentStatusEnum(str, Enum):
    PENDING = "pending"
    IN_TRANSIT = "in_transit"
    DELIVERED = "delivered"
    DELAYED = "delayed"
    CANCELLED = "cancelled"


class DelayReasonEnum(str, Enum):
    WEATHER = "weather"
    PORT_CONGESTION = "port_congestion"
    CUSTOMS = "customs"
    MECHANICAL = "mechanical"
    TRAFFIC = "traffic"
    OTHER = "other"


class CommunicationTypeEnum(str, Enum):
    EMAIL = "email"
    SMS = "sms"
    PHONE = "phone"
    WEBHOOK = "webhook"


# Base schemas
class BaseSchema(BaseModel):
    """Base schema with common configuration"""

    class Config:
        from_attributes = True
        use_enum_values = True


# Customer schemas
class CustomerBase(BaseSchema):
    name: str = Field(..., min_length=1, max_length=255)
    email: EmailStr
    phone: Optional[str] = Field(None, max_length=50)
    company: Optional[str] = Field(None, max_length=255)
    preferred_communication: Optional[CommunicationTypeEnum] = CommunicationTypeEnum.EMAIL
    language_preference: Optional[str] = Field("en", max_length=10)
    timezone: Optional[str] = Field("UTC", max_length=50)
    is_active: Optional[bool] = True


class CustomerCreate(CustomerBase):
    """Schema for creating a new customer"""
    pass


class CustomerUpdate(BaseSchema):
    """Schema for updating a customer"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    email: Optional[EmailStr] = None
    phone: Optional[str] = Field(None, max_length=50)
    company: Optional[str] = Field(None, max_length=255)
    preferred_communication: Optional[CommunicationTypeEnum] = None
    language_preference: Optional[str] = Field(None, max_length=10)
    timezone: Optional[str] = Field(None, max_length=50)
    is_active: Optional[bool] = None


class CustomerResponse(CustomerBase):
    """Schema for customer response"""
    id: int
    created_at: datetime
    updated_at: datetime


# Shipment schemas
class ShipmentBase(BaseSchema):
    shipment_id: str = Field(..., min_length=1, max_length=50)
    customer_id: int = Field(..., gt=0)
    origin_port: str = Field(..., min_length=1, max_length=100)
    destination_port: str = Field(..., min_length=1, max_length=100)
    route: Optional[str] = Field(None, max_length=500)
    container_type: Optional[str] = Field(None, max_length=50)
    cargo_weight: Optional[float] = Field(None, ge=0)
    cargo_value: Optional[float] = Field(None, ge=0)
    cargo_type: Optional[str] = Field(None, max_length=100)
    scheduled_departure: datetime
    scheduled_arrival: datetime
    is_priority: Optional[bool] = False


class ShipmentCreate(ShipmentBase):
    """Schema for creating a new shipment"""

    @validator('scheduled_arrival')
    def arrival_after_departure(cls, v, values):
        if 'scheduled_departure' in values and v <= values['scheduled_departure']:
            raise ValueError('Scheduled arrival must be after scheduled departure')
        return v


class ShipmentUpdate(BaseSchema):
    """Schema for updating a shipment"""
    origin_port: Optional[str] = Field(None, min_length=1, max_length=100)
    destination_port: Optional[str] = Field(None, min_length=1, max_length=100)
    route: Optional[str] = Field(None, max_length=500)
    container_type: Optional[str] = Field(None, max_length=50)
    cargo_weight: Optional[float] = Field(None, ge=0)
    cargo_value: Optional[float] = Field(None, ge=0)
    cargo_type: Optional[str] = Field(None, max_length=100)
    scheduled_departure: Optional[datetime] = None
    actual_departure: Optional[datetime] = None
    scheduled_arrival: Optional[datetime] = None
    estimated_arrival: Optional[datetime] = None
    actual_arrival: Optional[datetime] = None
    status: Optional[ShipmentStatusEnum] = None
    is_priority: Optional[bool] = None


class ShipmentResponse(ShipmentBase):
    """Schema for shipment response"""
    id: int
    actual_departure: Optional[datetime] = None
    estimated_arrival: Optional[datetime] = None
    actual_arrival: Optional[datetime] = None
    status: ShipmentStatusEnum
    created_at: datetime
    updated_at: datetime

    # Related data
    customer: Optional[CustomerResponse] = None


# Delay Prediction schemas
class DelayPredictionBase(BaseSchema):
    predicted_delay_hours: float = Field(..., ge=0)
    delay_probability: float = Field(..., ge=0.0, le=1.0)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    model_name: str = Field(..., min_length=1, max_length=100)
    model_version: str = Field(..., min_length=1, max_length=50)
    features_used: Optional[Dict[str, Any]] = None
    weather_factor: Optional[float] = None
    port_congestion_factor: Optional[float] = None
    route_complexity_factor: Optional[float] = None
    prediction_horizon_hours: Optional[int] = Field(24, ge=1, le=168)  # 1 hour to 1 week


class DelayPredictionCreate(DelayPredictionBase):
    """Schema for creating a delay prediction"""
    shipment_id: int = Field(..., gt=0)


class DelayPredictionResponse(DelayPredictionBase):
    """Schema for delay prediction response"""
    id: int
    shipment_id: int
    prediction_date: datetime
    actual_delay_hours: Optional[float] = None
    prediction_accuracy: Optional[float] = None


# Communication schemas
class CommunicationBase(BaseSchema):
    type: CommunicationTypeEnum
    subject: Optional[str] = Field(None, max_length=500)
    message: str = Field(..., min_length=1)
    recipient: str = Field(..., min_length=1, max_length=255)
    ai_generated: Optional[bool] = True
    model_used: Optional[str] = Field(None, max_length=100)
    template_used: Optional[str] = Field(None, max_length=100)
    personalization_data: Optional[Dict[str, Any]] = None


class CommunicationCreate(CommunicationBase):
    """Schema for creating a communication log"""
    shipment_id: int = Field(..., gt=0)
    customer_id: int = Field(..., gt=0)


class CommunicationResponse(CommunicationBase):
    """Schema for communication response"""
    id: int
    shipment_id: int
    customer_id: int
    sent_at: datetime
    delivered_at: Optional[datetime] = None
    read_at: Optional[datetime] = None
    response_at: Optional[datetime] = None
    delivery_status: Optional[str] = None
    engagement_score: Optional[float] = None


# Tracking Event schemas
class TrackingEventBase(BaseSchema):
    event_type: str = Field(..., min_length=1, max_length=100)
    event_description: Optional[str] = None
    event_location: Optional[str] = Field(None, max_length=255)
    event_timestamp: datetime
    event_data: Optional[Dict[str, Any]] = None
    data_source: Optional[str] = Field(None, max_length=100)


class TrackingEventCreate(TrackingEventBase):
    """Schema for creating a tracking event"""
    shipment_id: int = Field(..., gt=0)


class TrackingEventResponse(TrackingEventBase):
    """Schema for tracking event response"""
    id: int
    shipment_id: int
    created_at: datetime


# Prediction Request schemas
class PredictionRequest(BaseSchema):
    """Schema for delay prediction request"""
    shipment_id: Optional[int] = None
    origin_port: str = Field(..., min_length=1, max_length=100)
    destination_port: str = Field(..., min_length=1, max_length=100)
    scheduled_departure: datetime
    scheduled_arrival: datetime
    cargo_weight: Optional[float] = Field(None, ge=0)
    cargo_type: Optional[str] = Field(None, max_length=100)
    container_type: Optional[str] = Field(None, max_length=50)
    route: Optional[str] = Field(None, max_length=500)
    is_priority: Optional[bool] = False

    # External factors (optional, will be fetched if not provided)
    weather_conditions: Optional[Dict[str, Any]] = None
    port_congestion_data: Optional[Dict[str, Any]] = None
    traffic_conditions: Optional[Dict[str, Any]] = None


class PredictionResponse(BaseSchema):
    """Schema for prediction response"""
    shipment_id: Optional[int] = None
    predicted_delay_hours: float
    delay_probability: float
    confidence_score: float
    risk_level: str  # low, medium, high
    factors: Dict[str, float]
    recommendations: List[str]
    model_info: Dict[str, str]
    prediction_timestamp: datetime


# Communication Generation schemas
class CommunicationGenerationRequest(BaseSchema):
    """Schema for AI communication generation request"""
    shipment_id: int = Field(..., gt=0)
    customer_id: int = Field(..., gt=0)
    communication_type: CommunicationTypeEnum
    context: str = Field(..., min_length=1)  # delay, update, reminder, etc.
    delay_info: Optional[Dict[str, Any]] = None
    custom_instructions: Optional[str] = None
    language: Optional[str] = "en"
    tone: Optional[str] = "professional"  # professional, friendly, urgent, etc.


class CommunicationGenerationResponse(BaseSchema):
    """Schema for AI communication generation response"""
    subject: str
    message: str
    recipient: str
    type: CommunicationTypeEnum
    personalization_used: Dict[str, Any]
    model_used: str
    generated_at: datetime


# Dashboard schemas
class DashboardStats(BaseSchema):
    """Schema for dashboard statistics"""
    total_shipments: int
    active_shipments: int
    delayed_shipments: int
    on_time_shipments: int
    total_customers: int
    predictions_today: int
    communications_sent_today: int
    average_delay_hours: float
    delay_prediction_accuracy: float


class DashboardShipment(BaseSchema):
    """Schema for dashboard shipment display"""
    id: int
    shipment_id: str
    customer_name: str
    origin_port: str
    destination_port: str
    status: ShipmentStatusEnum
    scheduled_arrival: datetime
    estimated_arrival: Optional[datetime]
    delay_probability: Optional[float]
    predicted_delay_hours: Optional[float]
    risk_level: Optional[str]


# Error schemas
class ErrorResponse(BaseSchema):
    """Schema for error responses"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime
    path: Optional[str] = None


# Health check schema
class HealthResponse(BaseSchema):
    """Schema for health check response"""
    status: str
    timestamp: datetime
    version: str
    database: str
    redis: str
    external_apis: Dict[str, str]
