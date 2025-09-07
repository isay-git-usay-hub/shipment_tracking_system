"""
Core models for Maersk Shipment AI System
"""
from .base import BaseModel, Base
from .shipment import Shipment, ShipmentStatus, TrafficStatus, DelayReason
from .prediction import DelayPrediction

__all__ = [
    "BaseModel",
    "Base", 
    "Shipment",
    "ShipmentStatus",
    "TrafficStatus", 
    "DelayReason",
    "DelayPrediction",
]
