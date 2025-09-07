"""
Communication endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List

from models.database.connection import get_db
from models.schemas import (
    CommunicationGenerationRequest, CommunicationGenerationResponse,
    CommunicationResponse
)
from services.communication.service import communication_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/generate", response_model=CommunicationGenerationResponse)
async def generate_communication(request: CommunicationGenerationRequest):
    """Generate AI-powered communication"""
    try:
        communication = await communication_service.generate_communication(request)
        return communication

    except Exception as e:
        logger.error(f"Error generating communication: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating communication"
        )


@router.post("/send")
async def generate_and_send_communication(request: CommunicationGenerationRequest):
    """Generate and send communication"""
    try:
        result = await communication_service.generate_and_send_communication(request)
        return result

    except Exception as e:
        logger.error(f"Error generating and sending communication: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error sending communication"
        )


@router.post("/batch")
async def batch_generate_communications(requests: List[CommunicationGenerationRequest]):
    """Generate and send multiple communications"""
    try:
        results = await communication_service.batch_generate_communications(requests)
        return results

    except Exception as e:
        logger.error(f"Error in batch communication generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing batch communications"
        )


@router.get("/shipment/{shipment_id}", response_model=List[CommunicationResponse])
async def get_shipment_communications(
    shipment_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db)
):
    """Get communication history for a shipment"""
    try:
        from models.database.models import CommunicationLog, Shipment

        # Verify shipment exists
        shipment = db.query(Shipment).filter(Shipment.id == shipment_id).first()
        if not shipment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Shipment not found"
            )

        communications = db.query(CommunicationLog).filter(
            CommunicationLog.shipment_id == shipment_id
        ).order_by(CommunicationLog.sent_at.desc()).offset(skip).limit(limit).all()

        return communications

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching communications for shipment {shipment_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching communications"
        )


@router.get("/customer/{customer_id}", response_model=List[CommunicationResponse])
async def get_customer_communications(
    customer_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db)
):
    """Get communication history for a customer"""
    try:
        from models.database.models import CommunicationLog, Customer

        # Verify customer exists
        customer = db.query(Customer).filter(Customer.id == customer_id).first()
        if not customer:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Customer not found"
            )

        communications = db.query(CommunicationLog).filter(
            CommunicationLog.customer_id == customer_id
        ).order_by(CommunicationLog.sent_at.desc()).offset(skip).limit(limit).all()

        return communications

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching communications for customer {customer_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching communications"
        )


@router.get("/stats")
async def get_communication_stats(
    days_back: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """Get communication statistics"""
    try:
        from models.database.models import CommunicationLog
        from datetime import datetime, timedelta
        from sqlalchemy import func

        cutoff_date = datetime.now() - timedelta(days=days_back)

        # Total communications
        total_comms = db.query(CommunicationLog).filter(
            CommunicationLog.sent_at >= cutoff_date
        ).count()

        # Communications by type
        comm_by_type = db.query(
            CommunicationLog.type,
            func.count(CommunicationLog.id).label('count')
        ).filter(
            CommunicationLog.sent_at >= cutoff_date
        ).group_by(CommunicationLog.type).all()

        # Delivery success rate
        successful_deliveries = db.query(CommunicationLog).filter(
            CommunicationLog.sent_at >= cutoff_date,
            CommunicationLog.delivery_status == 'delivered'
        ).count()

        success_rate = (successful_deliveries / total_comms * 100) if total_comms > 0 else 0

        # AI generated percentage
        ai_generated = db.query(CommunicationLog).filter(
            CommunicationLog.sent_at >= cutoff_date,
            CommunicationLog.ai_generated == True
        ).count()

        ai_percentage = (ai_generated / total_comms * 100) if total_comms > 0 else 0

        return {
            "period_days": days_back,
            "total_communications": total_comms,
            "communications_by_type": dict(comm_by_type),
            "delivery_success_rate": round(success_rate, 2),
            "ai_generated_percentage": round(ai_percentage, 2),
            "successful_deliveries": successful_deliveries
        }

    except Exception as e:
        logger.error(f"Error calculating communication stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error calculating statistics"
        )
