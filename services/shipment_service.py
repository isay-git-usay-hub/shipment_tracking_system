"""
Shipment service with business logic
"""
import logging
from datetime import datetime, timedelta
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, func

from core.models import Shipment, ShipmentStatus, TrafficStatus, DelayReason
from core.schemas import (
    ShipmentCreate, 
    ShipmentUpdate, 
    ShipmentResponse,
    ShipmentSummary,
    ShipmentFilter,
)

logger = logging.getLogger(__name__)


class ShipmentService:
    """Service class for shipment operations"""
    
    def create_shipment(self, db: Session, shipment_data: ShipmentCreate) -> Shipment:
        """Create a new shipment"""
        try:
            # Convert schema enums to model enums
            shipment_dict = shipment_data.model_dump()
            
            # Convert enum values
            if shipment_dict.get('shipment_status'):
                shipment_dict['shipment_status'] = ShipmentStatus(shipment_dict['shipment_status'])
            if shipment_dict.get('traffic_status'):
                shipment_dict['traffic_status'] = TrafficStatus(shipment_dict['traffic_status'])
            if shipment_dict.get('logistics_delay_reason'):
                shipment_dict['logistics_delay_reason'] = DelayReason(shipment_dict['logistics_delay_reason'])
            
            shipment = Shipment(**shipment_dict)
            db.add(shipment)
            db.commit()
            db.refresh(shipment)
            
            logger.info(f"Created shipment {shipment.id} for asset {shipment.asset_id}")
            return shipment
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating shipment: {e}")
            raise
    
    def get_shipment(self, db: Session, shipment_id: int) -> Optional[Shipment]:
        """Get a shipment by ID"""
        return db.query(Shipment).filter(
            and_(Shipment.id == shipment_id, Shipment.is_active == True)
        ).first()
    
    def get_shipments(
        self, 
        db: Session, 
        filters: Optional[ShipmentFilter] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Shipment]:
        """Get shipments with optional filtering"""
        query = db.query(Shipment).filter(Shipment.is_active == True)
        
        if filters:
            if filters.asset_id:
                query = query.filter(Shipment.asset_id == filters.asset_id)
            
            if filters.shipment_status:
                query = query.filter(Shipment.shipment_status == ShipmentStatus(filters.shipment_status))
            
            if filters.logistics_delay is not None:
                query = query.filter(Shipment.logistics_delay == filters.logistics_delay)
            
            if filters.traffic_status:
                query = query.filter(Shipment.traffic_status == TrafficStatus(filters.traffic_status))
            
            if filters.delay_reason:
                query = query.filter(Shipment.logistics_delay_reason == DelayReason(filters.delay_reason))
            
            if filters.start_date:
                query = query.filter(Shipment.timestamp >= filters.start_date)
            
            if filters.end_date:
                query = query.filter(Shipment.timestamp <= filters.end_date)
        
        return query.order_by(desc(Shipment.timestamp)).offset(skip).limit(limit).all()
    
    def update_shipment(
        self, 
        db: Session, 
        shipment_id: int, 
        shipment_update: ShipmentUpdate
    ) -> Optional[Shipment]:
        """Update a shipment"""
        try:
            shipment = self.get_shipment(db, shipment_id)
            if not shipment:
                return None
            
            update_data = shipment_update.model_dump(exclude_unset=True)
            
            # Convert enum values
            if 'shipment_status' in update_data:
                update_data['shipment_status'] = ShipmentStatus(update_data['shipment_status'])
            if 'traffic_status' in update_data:
                update_data['traffic_status'] = TrafficStatus(update_data['traffic_status'])
            if 'logistics_delay_reason' in update_data:
                update_data['logistics_delay_reason'] = DelayReason(update_data['logistics_delay_reason'])
            
            for field, value in update_data.items():
                setattr(shipment, field, value)
            
            db.commit()
            db.refresh(shipment)
            
            logger.info(f"Updated shipment {shipment_id}")
            return shipment
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error updating shipment {shipment_id}: {e}")
            raise
    
    def delete_shipment(self, db: Session, shipment_id: int) -> bool:
        """Soft delete a shipment"""
        try:
            shipment = self.get_shipment(db, shipment_id)
            if not shipment:
                return False
            
            shipment.is_active = False
            db.commit()
            
            logger.info(f"Deleted shipment {shipment_id}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error deleting shipment {shipment_id}: {e}")
            raise
    
    def get_delayed_shipments(self, db: Session, hours_back: int = 24) -> List[Shipment]:
        """Get shipments that are currently delayed"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        return db.query(Shipment).filter(
            and_(
                Shipment.is_active == True,
                Shipment.logistics_delay == True,
                Shipment.timestamp >= cutoff_time
            )
        ).order_by(desc(Shipment.timestamp)).all()
    
    def get_shipment_statistics(self, db: Session) -> dict:
        """Get overall shipment statistics"""
        try:
            # Total shipments
            total_shipments = db.query(Shipment).filter(Shipment.is_active == True).count()
            
            # Delayed shipments
            delayed_shipments = db.query(Shipment).filter(
                and_(Shipment.is_active == True, Shipment.logistics_delay == True)
            ).count()
            
            # Status distribution
            status_counts = db.query(
                Shipment.shipment_status,
                func.count(Shipment.id).label('count')
            ).filter(Shipment.is_active == True).group_by(Shipment.shipment_status).all()
            
            # Delay reasons distribution
            delay_reasons = db.query(
                Shipment.logistics_delay_reason,
                func.count(Shipment.id).label('count')
            ).filter(
                and_(Shipment.is_active == True, Shipment.logistics_delay == True)
            ).group_by(Shipment.logistics_delay_reason).all()
            
            # Average waiting time
            avg_waiting_time = db.query(func.avg(Shipment.waiting_time)).filter(
                Shipment.is_active == True
            ).scalar() or 0
            
            return {
                "total_shipments": total_shipments,
                "delayed_shipments": delayed_shipments,
                "delay_rate": (delayed_shipments / total_shipments * 100) if total_shipments > 0 else 0,
                "status_distribution": {str(status): count for status, count in status_counts},
                "delay_reasons": {str(reason): count for reason, count in delay_reasons},
                "avg_waiting_time_minutes": round(avg_waiting_time, 2),
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting shipment statistics: {e}")
            raise


# Global service instance
shipment_service = ShipmentService()
