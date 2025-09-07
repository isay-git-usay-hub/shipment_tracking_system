"""
Pydantic schemas for shipment data
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class ShipmentStatusEnum(str, Enum):
    """Shipment status enumeration"""
    IN_TRANSIT = "In Transit"
    DELIVERED = "Delivered"
    DELAYED = "Delayed"
    PENDING = "Pending"
    CANCELLED = "Cancelled"


class TrafficStatusEnum(str, Enum):
    """Traffic status enumeration"""
    CLEAR = "Clear"
    HEAVY = "Heavy"
    DETOUR = "Detour"


class DelayReasonEnum(str, Enum):
    """Delay reason enumeration"""
    NONE = "None"
    WEATHER = "Weather"
    TRAFFIC = "Traffic"
    MECHANICAL_FAILURE = "Mechanical Failure"
    OTHER = "Other"


class ShipmentBase(BaseModel):
    """Base shipment schema"""
    asset_id: str = Field(..., description="Asset identifier")
    latitude: float = Field(..., description="Latitude coordinate")
    longitude: float = Field(..., description="Longitude coordinate")
    inventory_level: int = Field(..., description="Current inventory level")
    shipment_status: ShipmentStatusEnum = Field(..., description="Current shipment status")
    temperature: Optional[float] = Field(None, description="Temperature in Celsius")
    humidity: Optional[float] = Field(None, description="Humidity percentage")
    traffic_status: Optional[TrafficStatusEnum] = Field(None, description="Traffic conditions")
    waiting_time: int = Field(..., description="Waiting time in minutes")
    user_transaction_amount: Optional[float] = Field(None, description="Transaction amount")
    user_purchase_frequency: Optional[int] = Field(None, description="Purchase frequency")
    logistics_delay_reason: Optional[DelayReasonEnum] = Field(None, description="Reason for delay")
    asset_utilization: Optional[float] = Field(None, description="Asset utilization percentage")
    demand_forecast: Optional[float] = Field(None, description="Demand forecast value")
    logistics_delay: bool = Field(..., description="Whether shipment is delayed")
    timestamp: datetime = Field(..., description="Timestamp of the record")


class ShipmentCreate(ShipmentBase):
    """Schema for creating a new shipment"""
    pass


class ShipmentUpdate(BaseModel):
    """Schema for updating a shipment"""
    shipment_status: Optional[ShipmentStatusEnum] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    traffic_status: Optional[TrafficStatusEnum] = None
    waiting_time: Optional[int] = None
    logistics_delay_reason: Optional[DelayReasonEnum] = None
    logistics_delay: Optional[bool] = None


class ShipmentResponse(ShipmentBase):
    """Schema for shipment response"""
    id: int
    created_at: datetime
    updated_at: datetime
    is_active: bool
    delay_probability: float = Field(..., description="Calculated delay probability")
    estimated_delay_hours: float = Field(..., description="Estimated delay in hours")
    
    model_config = ConfigDict(from_attributes=True)


class ShipmentSummary(BaseModel):
    """Summary schema for shipment lists"""
    id: int
    asset_id: str
    shipment_status: ShipmentStatusEnum
    latitude: float
    longitude: float
    logistics_delay: bool
    delay_probability: float
    timestamp: datetime
    
    model_config = ConfigDict(from_attributes=True)


class ShipmentFilter(BaseModel):
    """Schema for filtering shipments"""
    asset_id: Optional[str] = None
    shipment_status: Optional[ShipmentStatusEnum] = None
    logistics_delay: Optional[bool] = None
    traffic_status: Optional[TrafficStatusEnum] = None
    delay_reason: Optional[DelayReasonEnum] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: int = Field(default=100, le=50000)
    offset: int = Field(default=0, ge=0)
