"""
Shipment models for Maersk Shipment AI System
Based on smart_logistics_dataset.csv structure
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Enum, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum

from .base import BaseModel


class ShipmentStatus(enum.Enum):
    """Shipment status from the dataset"""
    IN_TRANSIT = "In Transit"
    DELIVERED = "Delivered"
    DELAYED = "Delayed"
    PENDING = "Pending"
    CANCELLED = "Cancelled"


class TrafficStatus(enum.Enum):
    """Traffic status from the dataset"""
    CLEAR = "Clear"
    HEAVY = "Heavy"
    DETOUR = "Detour"


class DelayReason(enum.Enum):
    """Delay reasons from the dataset"""
    NONE = "None"
    WEATHER = "Weather"
    TRAFFIC = "Traffic"
    MECHANICAL_FAILURE = "Mechanical Failure"
    OTHER = "Other"


class Shipment(BaseModel):
    """
    Shipment model based on smart_logistics_dataset.csv
    """
    __tablename__ = "shipments"
    
    # Asset information
    asset_id = Column(String(50), nullable=False, index=True)
    
    # Location data
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    
    # Logistics data
    inventory_level = Column(Integer, nullable=False)
    shipment_status = Column(Enum(ShipmentStatus), nullable=False, index=True)
    temperature = Column(Float)
    humidity = Column(Float)
    traffic_status = Column(Enum(TrafficStatus))
    waiting_time = Column(Integer, nullable=False)  # in minutes
    
    # Business data
    user_transaction_amount = Column(Float)
    user_purchase_frequency = Column(Integer)
    logistics_delay_reason = Column(Enum(DelayReason))
    asset_utilization = Column(Float)
    demand_forecast = Column(Float)
    
    # Target variable
    logistics_delay = Column(Boolean, nullable=False, index=True)  # 1 = delayed, 0 = on time
    
    # Metadata
    timestamp = Column(DateTime, nullable=False)
    
    # Relationships
    predictions = relationship("DelayPrediction", back_populates="shipment")
    
    # Indexes for better query performance
    __table_args__ = (
        Index('idx_shipment_status_timestamp', 'shipment_status', 'timestamp'),
        Index('idx_shipment_delay', 'logistics_delay'),
        Index('idx_shipment_asset', 'asset_id', 'timestamp'),
        Index('idx_shipment_location', 'latitude', 'longitude'),
    )
    
    @property
    def delay_probability(self) -> float:
        """Calculate delay probability using ML model or heuristic fallback"""
        try:
            # Try to use ML model
            import sys
            import os
            ml_path = os.path.join(os.path.dirname(__file__), '..', '..', 'ml')
            sys.path.insert(0, ml_path)
            from delay_predictor import get_delay_predictor
            import pandas as pd
            
            # Create DataFrame with single row
            data = {
                'Asset_ID': [self.asset_id],
                'Latitude': [self.latitude],
                'Longitude': [self.longitude],
                'Inventory_Level': [self.inventory_level],
                'Shipment_Status': [self.shipment_status.value],
                'Temperature': [self.temperature],
                'Humidity': [self.humidity],
                'Traffic_Status': [self.traffic_status.value if self.traffic_status else 'Clear'],
                'Waiting_Time': [self.waiting_time],
                'User_Transaction_Amount': [self.user_transaction_amount or 0],
                'User_Purchase_Frequency': [self.user_purchase_frequency or 0],
                'Logistics_Delay_Reason': [self.logistics_delay_reason.value if self.logistics_delay_reason else 'None'],
                'Asset_Utilization': [self.asset_utilization or 0],
                'Demand_Forecast': [self.demand_forecast or 0],
                'Timestamp': [self.timestamp]
            }
            
            df = pd.DataFrame(data)
            predictor = get_delay_predictor()
            _, probabilities = predictor.predict(df)
            
            return float(probabilities[0])
            
        except Exception:
            # Fallback to heuristic
            factors = 0
            total_factors = 0
            
            if self.traffic_status == TrafficStatus.HEAVY:
                factors += 0.3
            elif self.traffic_status == TrafficStatus.DETOUR:
                factors += 0.2
            total_factors += 1
            
            if self.waiting_time > 30:
                factors += 0.4
            total_factors += 1
            
            if self.logistics_delay_reason != DelayReason.NONE:
                factors += 0.5
            total_factors += 1
            
            return min(factors / total_factors if total_factors > 0 else 0.0, 1.0)
    
    @property
    def estimated_delay_hours(self) -> float:
        """Estimate delay in hours based on current conditions"""
        base_delay = self.waiting_time / 60.0  # Convert minutes to hours
        
        if self.logistics_delay_reason == DelayReason.WEATHER:
            base_delay *= 2.0
        elif self.logistics_delay_reason == DelayReason.MECHANICAL_FAILURE:
            base_delay *= 3.0
        elif self.logistics_delay_reason == DelayReason.TRAFFIC:
            base_delay *= 1.5
        
        return max(base_delay, 0.0)
