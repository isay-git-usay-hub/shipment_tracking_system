"""
Shipment management endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from models.database.connection import get_db
from models.database.models import Shipment, Customer
from models.schemas import (
    ShipmentCreate, ShipmentUpdate, ShipmentResponse,
    TrackingEventCreate, TrackingEventResponse
)
from services.prediction.service import prediction_service
from services.communication.service import communication_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/", response_model=ShipmentResponse, status_code=status.HTTP_201_CREATED)
async def create_shipment(
    shipment: ShipmentCreate, 
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Create a new shipment"""
    try:
        # Verify customer exists
        customer = db.query(Customer).filter(Customer.id == shipment.customer_id).first()
        if not customer:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Customer not found"
            )

        # Check if shipment ID already exists
        existing = db.query(Shipment).filter(Shipment.shipment_id == shipment.shipment_id).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Shipment ID already exists"
            )

        db_shipment = Shipment(**shipment.model_dump())
        db.add(db_shipment)
        db.commit()
        db.refresh(db_shipment)

        # Add background task to generate initial prediction
        background_tasks.add_task(generate_initial_prediction, db_shipment.id)

        logger.info(f"Created shipment: {db_shipment.id}")
        return db_shipment

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating shipment: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error creating shipment"
        )


@router.get("/", response_model=List[ShipmentResponse])
async def get_shipments(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status_filter: Optional[str] = Query(None),
    customer_id: Optional[int] = Query(None),
    db: Session = Depends(get_db)
):
    """Get shipments with filtering and pagination"""
    try:
        query = db.query(Shipment)

        if status_filter:
            query = query.filter(Shipment.status == status_filter)

        if customer_id:
            query = query.filter(Shipment.customer_id == customer_id)

        shipments = query.order_by(Shipment.created_at.desc()).offset(skip).limit(limit).all()
        return shipments

    except Exception as e:
        logger.error(f"Error fetching shipments: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching shipments"
        )


@router.get("/{shipment_id}", response_model=ShipmentResponse)
async def get_shipment(shipment_id: int, db: Session = Depends(get_db)):
    """Get shipment by ID"""
    try:
        shipment = db.query(Shipment).filter(Shipment.id == shipment_id).first()
        if not shipment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Shipment not found"
            )
        return shipment

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching shipment {shipment_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching shipment"
        )


@router.get("/tracking/{shipment_tracking_id}")
async def track_shipment(shipment_tracking_id: str, db: Session = Depends(get_db)):
    """Track shipment by shipment ID"""
    try:
        shipment = db.query(Shipment).filter(Shipment.shipment_id == shipment_tracking_id).first()
        if not shipment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Shipment not found"
            )

        from models.database.models import TrackingEvent, DelayPrediction

        # Get tracking events
        tracking_events = db.query(TrackingEvent).filter(
            TrackingEvent.shipment_id == shipment.id
        ).order_by(TrackingEvent.event_timestamp.desc()).all()

        # Get latest prediction
        latest_prediction = db.query(DelayPrediction).filter(
            DelayPrediction.shipment_id == shipment.id
        ).order_by(DelayPrediction.prediction_date.desc()).first()

        return {
            "shipment": shipment,
            "tracking_events": tracking_events,
            "delay_prediction": latest_prediction,
            "last_updated": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error tracking shipment {shipment_tracking_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error tracking shipment"
        )


@router.put("/{shipment_id}", response_model=ShipmentResponse)
async def update_shipment(
    shipment_id: int,
    shipment_update: ShipmentUpdate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Update shipment"""
    try:
        shipment = db.query(Shipment).filter(Shipment.id == shipment_id).first()
        if not shipment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Shipment not found"
            )

        # Update fields
        update_data = shipment_update.model_dump(exclude_unset=True)
        old_status = shipment.status

        for field, value in update_data.items():
            setattr(shipment, field, value)

        db.commit()
        db.refresh(shipment)

        # If status changed, add background tasks
        if 'status' in update_data and update_data['status'] != old_status:
            background_tasks.add_task(handle_status_change, shipment.id, old_status, update_data['status'])

        logger.info(f"Updated shipment: {shipment_id}")
        return shipment

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating shipment {shipment_id}: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error updating shipment"
        )


@router.post("/{shipment_id}/tracking", response_model=TrackingEventResponse)
async def add_tracking_event(
    shipment_id: int,
    event: TrackingEventCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Add tracking event to shipment"""
    try:
        # Verify shipment exists
        shipment = db.query(Shipment).filter(Shipment.id == shipment_id).first()
        if not shipment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Shipment not found"
            )

        from models.database.models import TrackingEvent

        db_event = TrackingEvent(**event.model_dump(), shipment_id=shipment_id)
        db.add(db_event)
        db.commit()
        db.refresh(db_event)

        # Add background task to check if customer notification needed
        background_tasks.add_task(check_notification_needed, shipment_id, db_event.event_type)

        logger.info(f"Added tracking event for shipment: {shipment_id}")
        return db_event

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding tracking event: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error adding tracking event"
        )


@router.get("/{shipment_id}/tracking", response_model=List[TrackingEventResponse])
async def get_shipment_tracking(shipment_id: int, db: Session = Depends(get_db)):
    """Get tracking events for shipment"""
    try:
        # Verify shipment exists
        shipment = db.query(Shipment).filter(Shipment.id == shipment_id).first()
        if not shipment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Shipment not found"
            )

        from models.database.models import TrackingEvent

        events = db.query(TrackingEvent).filter(
            TrackingEvent.shipment_id == shipment_id
        ).order_by(TrackingEvent.event_timestamp.desc()).all()

        return events

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching tracking events for shipment {shipment_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching tracking events"
        )


@router.post("/{shipment_id}/predict")
async def trigger_prediction(shipment_id: int, db: Session = Depends(get_db)):
    """Manually trigger delay prediction for shipment"""
    try:
        shipment = db.query(Shipment).filter(Shipment.id == shipment_id).first()
        if not shipment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Shipment not found"
            )

        from models.schemas import PredictionRequest

        # Create prediction request
        request = PredictionRequest(
            shipment_id=shipment.id,
            origin_port=shipment.origin_port,
            destination_port=shipment.destination_port,
            scheduled_departure=shipment.scheduled_departure,
            scheduled_arrival=shipment.scheduled_arrival,
            cargo_weight=shipment.cargo_weight,
            cargo_type=shipment.cargo_type,
            container_type=shipment.container_type,
            route=shipment.route,
            is_priority=shipment.is_priority
        )

        # Generate prediction
        prediction = await prediction_service.predict_delay(request)

        logger.info(f"Generated prediction for shipment: {shipment_id}")
        return prediction

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating prediction for shipment {shipment_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating prediction"
        )


# Background task functions
async def generate_initial_prediction(shipment_id: int):
    """Generate initial prediction for new shipment"""
    try:
        db = next(get_db())
        shipment = db.query(Shipment).filter(Shipment.id == shipment_id).first()
        if shipment:
            from models.schemas import PredictionRequest

            request = PredictionRequest(
                shipment_id=shipment.id,
                origin_port=shipment.origin_port,
                destination_port=shipment.destination_port,
                scheduled_departure=shipment.scheduled_departure,
                scheduled_arrival=shipment.scheduled_arrival,
                cargo_weight=shipment.cargo_weight,
                cargo_type=shipment.cargo_type,
                container_type=shipment.container_type,
                route=shipment.route,
                is_priority=shipment.is_priority
            )

            await prediction_service.predict_delay(request)
            logger.info(f"Generated initial prediction for shipment: {shipment_id}")

        db.close()
    except Exception as e:
        logger.error(f"Error in background prediction generation: {e}")


async def handle_status_change(shipment_id: int, old_status: str, new_status: str):
    """Handle shipment status changes"""
    try:
        # If shipment is delayed, trigger communication
        if new_status == 'delayed':
            from models.schemas import CommunicationGenerationRequest, CommunicationTypeEnum

            db = next(get_db())
            shipment = db.query(Shipment).filter(Shipment.id == shipment_id).first()
            if shipment and shipment.customer:
                request = CommunicationGenerationRequest(
                    shipment_id=shipment.id,
                    customer_id=shipment.customer_id,
                    communication_type=CommunicationTypeEnum.EMAIL,
                    context="delay_notification",
                    language="en",
                    tone="professional"
                )

                await communication_service.generate_and_send_communication(request)
                logger.info(f"Sent delay notification for shipment: {shipment_id}")

            db.close()

    except Exception as e:
        logger.error(f"Error handling status change: {e}")


async def check_notification_needed(shipment_id: int, event_type: str):
    """Check if customer notification is needed for tracking event"""
    try:
        # Determine if this event type requires customer notification
        notification_events = ['departed', 'arrived', 'delayed', 'customs_cleared']

        if event_type.lower() in notification_events:
            from models.schemas import CommunicationGenerationRequest, CommunicationTypeEnum

            db = next(get_db())
            shipment = db.query(Shipment).filter(Shipment.id == shipment_id).first()
            if shipment and shipment.customer:
                request = CommunicationGenerationRequest(
                    shipment_id=shipment.id,
                    customer_id=shipment.customer_id,
                    communication_type=CommunicationTypeEnum.EMAIL,
                    context="shipment_update",
                    language="en",
                    tone="professional"
                )

                await communication_service.generate_and_send_communication(request)
                logger.info(f"Sent tracking update for shipment: {shipment_id}")

            db.close()

    except Exception as e:
        logger.error(f"Error checking notification needs: {e}")
