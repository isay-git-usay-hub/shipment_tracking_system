"""
Database models for Maersk Shipment AI System
"""
from datetime import datetime, timedelta
from typing import Optional, List
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, Text, 
    ForeignKey, JSON, Enum as SQLEnum, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import enum
from sqlalchemy.sql import func

Base = declarative_base()


class ShipmentStatus(enum.Enum):
    """Shipment status enumeration"""
    PENDING = "pending"
    IN_TRANSIT = "in_transit"
    DELIVERED = "delivered"
    DELAYED = "delayed"
    CANCELLED = "cancelled"


class DelayReason(enum.Enum):
    """Delay reason enumeration"""
    WEATHER = "weather"
    PORT_CONGESTION = "port_congestion"
    CUSTOMS = "customs"
    MECHANICAL = "mechanical"
    TRAFFIC = "traffic"
    OTHER = "other"


class CommunicationType(enum.Enum):
    """Communication type enumeration"""
    EMAIL = "email"
    SMS = "sms"
    PHONE = "phone"
    WEBHOOK = "webhook"


class Customer(Base):
    """Customer model"""
    __tablename__ = "customers"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False, unique=True, index=True)
    phone = Column(String(50))
    company = Column(String(255))
    preferred_communication = Column(SQLEnum(CommunicationType), default=CommunicationType.EMAIL)
    language_preference = Column(String(10), default="en")
    timezone = Column(String(50), default="UTC")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    shipments = relationship("Shipment", back_populates="customer")
    communications = relationship("CommunicationLog", back_populates="customer")


class Shipment(Base):
    """Shipment model"""
    __tablename__ = "shipments"

    id = Column(Integer, primary_key=True, index=True)
    shipment_id = Column(String(50), unique=True, nullable=False, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)

    # Route Information
    origin_port = Column(String(100), nullable=False)
    destination_port = Column(String(100), nullable=False)
    route = Column(String(500))

    # Cargo Information
    container_type = Column(String(50))
    cargo_weight = Column(Float)
    cargo_value = Column(Float)
    cargo_type = Column(String(100))

    # Schedule Information
    scheduled_departure = Column(DateTime, nullable=False)
    actual_departure = Column(DateTime)
    scheduled_arrival = Column(DateTime, nullable=False)
    estimated_arrival = Column(DateTime)
    actual_arrival = Column(DateTime)

    # Status Information
    status = Column(SQLEnum(ShipmentStatus), default=ShipmentStatus.PENDING, index=True)
    is_priority = Column(Boolean, default=False)

    # Metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    customer = relationship("Customer", back_populates="shipments")
    delay_predictions = relationship("DelayPrediction", back_populates="shipment")
    communications = relationship("CommunicationLog", back_populates="shipment")
    tracking_events = relationship("TrackingEvent", back_populates="shipment")

    # Indexes
    __table_args__ = (
        Index('idx_shipment_status_departure', 'status', 'scheduled_departure'),
        Index('idx_shipment_customer_status', 'customer_id', 'status'),
    )


class DelayPrediction(Base):
    """Delay prediction model"""
    __tablename__ = "delay_predictions"

    id = Column(Integer, primary_key=True, index=True)
    shipment_id = Column(Integer, ForeignKey("shipments.id"), nullable=False)

    # Prediction Information
    predicted_delay_hours = Column(Float, nullable=False)
    delay_probability = Column(Float, nullable=False)  # 0.0 to 1.0
    confidence_score = Column(Float, nullable=False)   # 0.0 to 1.0

    # Model Information
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    features_used = Column(JSON)  # Store feature values used for prediction

    # External Factors
    weather_factor = Column(Float)
    port_congestion_factor = Column(Float)
    route_complexity_factor = Column(Float)

    # Prediction Metadata
    prediction_date = Column(DateTime, default=func.now())
    prediction_horizon_hours = Column(Integer, default=24)  # How far ahead we're predicting

    # Validation
    actual_delay_hours = Column(Float)  # Filled after actual arrival
    prediction_accuracy = Column(Float)  # Calculated after validation

    # Relationships
    shipment = relationship("Shipment", back_populates="delay_predictions")


class CommunicationLog(Base):
    """Communication log model"""
    __tablename__ = "communication_logs"

    id = Column(Integer, primary_key=True, index=True)
    shipment_id = Column(Integer, ForeignKey("shipments.id"), nullable=False)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)

    # Communication Details
    type = Column(SQLEnum(CommunicationType), nullable=False)
    subject = Column(String(500))
    message = Column(Text, nullable=False)
    recipient = Column(String(255), nullable=False)

    # Status
    sent_at = Column(DateTime, default=func.now())
    delivered_at = Column(DateTime)
    read_at = Column(DateTime)
    response_at = Column(DateTime)

    # AI Generation Details
    ai_generated = Column(Boolean, default=True)
    model_used = Column(String(100))
    template_used = Column(String(100))
    personalization_data = Column(JSON)

    # Success Metrics
    delivery_status = Column(String(50))  # sent, delivered, failed, etc.
    engagement_score = Column(Float)  # Click-through, response rate, etc.

    # Relationships
    shipment = relationship("Shipment", back_populates="communications")
    customer = relationship("Customer", back_populates="communications")


class TrackingEvent(Base):
    """Tracking event model"""
    __tablename__ = "tracking_events"

    id = Column(Integer, primary_key=True, index=True)
    shipment_id = Column(Integer, ForeignKey("shipments.id"), nullable=False)

    # Event Information
    event_type = Column(String(100), nullable=False)  # departed, arrived, delayed, etc.
    event_description = Column(Text)
    event_location = Column(String(255))
    event_timestamp = Column(DateTime, nullable=False)

    # Additional Data
    event_data = Column(JSON)  # Any additional structured data

    # Source Information
    data_source = Column(String(100))  # API, manual, sensor, etc.
    created_at = Column(DateTime, default=func.now())

    # Relationships
    shipment = relationship("Shipment", back_populates="tracking_events")

    # Indexes
    __table_args__ = (
        Index('idx_tracking_shipment_timestamp', 'shipment_id', 'event_timestamp'),
    )


class ModelPerformance(Base):
    """Model performance tracking"""
    __tablename__ = "model_performance"

    id = Column(Integer, primary_key=True, index=True)

    # Model Information
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)

    # Performance Metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    rmse = Column(Float)
    mae = Column(Float)

    # Training Information
    training_date = Column(DateTime, nullable=False)
    training_data_size = Column(Integer)
    training_duration_minutes = Column(Integer)

    # Validation Information
    validation_start_date = Column(DateTime)
    validation_end_date = Column(DateTime)
    predictions_count = Column(Integer, default=0)
    successful_predictions = Column(Integer, default=0)

    # Metadata
    hyperparameters = Column(JSON)
    feature_importance = Column(JSON)
    created_at = Column(DateTime, default=func.now())


class ExternalData(Base):
    """External data cache"""
    __tablename__ = "external_data"

    id = Column(Integer, primary_key=True, index=True)

    # Data Identification
    data_type = Column(String(100), nullable=False)  # weather, port_status, traffic, etc.
    data_key = Column(String(255), nullable=False)   # location, route, etc.

    # Data Content
    data_value = Column(JSON, nullable=False)

    # Cache Information
    created_at = Column(DateTime, default=func.now())
    expires_at = Column(DateTime, nullable=False)
    last_updated = Column(DateTime, default=func.now())

    # Source Information
    source = Column(String(100), nullable=False)
    source_url = Column(String(500))

    # Indexes
    __table_args__ = (
        Index('idx_external_data_type_key', 'data_type', 'data_key'),
        Index('idx_external_data_expires', 'expires_at'),
    )
