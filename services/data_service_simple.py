"""
Simplified data service for loading smart logistics dataset
"""
import logging
import csv
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
from sqlalchemy.orm import Session

from core.models import Shipment, ShipmentStatus, TrafficStatus, DelayReason
from core.database import get_db_session

logger = logging.getLogger(__name__)


class DataService:
    """Simplified service for data loading and processing"""
    
    def __init__(self):
        self.data_path = Path("data")
        self.dataset_file = self.data_path / "smart_logistics_dataset.csv"
    
    def get_dataset_summary(self) -> Dict[str, Any]:
        """Get basic dataset summary"""
        try:
            if not self.dataset_file.exists():
                return {
                    "status": "Dataset file not found",
                    "total_records": 0,
                    "file_path": str(self.dataset_file)
                }
            
            # Count records
            record_count = 0
            with open(self.dataset_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for _ in reader:
                    record_count += 1
            
            return {
                "status": "Dataset available",
                "total_records": record_count,
                "file_path": str(self.dataset_file),
                "file_size_mb": round(self.dataset_file.stat().st_size / 1024 / 1024, 2),
                "last_modified": datetime.fromtimestamp(self.dataset_file.stat().st_mtime).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting dataset summary: {e}")
            return {
                "status": "Error loading dataset",
                "error": str(e),
                "total_records": 0
            }
    
    def validate_database_data(self, db: Session) -> Dict[str, Any]:
        """Validate data integrity in database"""
        try:
            total_shipments = db.query(Shipment).filter(Shipment.is_active == True).count()
            delayed_shipments = db.query(Shipment).filter(
                Shipment.is_active == True, 
                Shipment.logistics_delay == True
            ).count()
            
            validation_result = {
                "total_records": total_shipments,
                "delayed_records": delayed_shipments,
                "delay_rate": (delayed_shipments / total_shipments * 100) if total_shipments > 0 else 0,
                "data_integrity": "VALID" if total_shipments > 0 else "NO_DATA",
                "validation_timestamp": datetime.utcnow().isoformat()
            }
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating database data: {e}")
            return {
                "status": "Error validating data",
                "error": str(e),
                "data_integrity": "ERROR"
            }
    
    def load_data_to_database(self, batch_size: int = 100, replace_existing: bool = False) -> int:
        """Load sample data for testing"""
        try:
            with get_db_session() as db:
                # Check if data already exists
                existing_count = db.query(Shipment).count()
                if existing_count > 0 and not replace_existing:
                    logger.info(f"Database already contains {existing_count} shipments. Use replace_existing=True to reload.")
                    return existing_count
                
                if replace_existing and existing_count > 0:
                    logger.info(f"Removing {existing_count} existing shipments")
                    db.query(Shipment).delete()
                    db.commit()
                
                # Create sample data for testing
                sample_shipments = []
                
                # Create 10 sample shipments
                for i in range(10):
                    shipment = Shipment(
                        asset_id=f"Truck_{i+1}",
                        latitude=40.7128 + (i * 0.1),
                        longitude=-74.0060 + (i * 0.1),
                        inventory_level=100 + (i * 10),
                        shipment_status=ShipmentStatus.IN_TRANSIT if i % 3 == 0 else 
                                      (ShipmentStatus.DELAYED if i % 3 == 1 else ShipmentStatus.DELIVERED),
                        temperature=20.0 + (i * 0.5),
                        humidity=60.0 + (i * 2.0),
                        traffic_status=TrafficStatus.CLEAR if i % 2 == 0 else TrafficStatus.HEAVY,
                        waiting_time=10 + (i * 5),
                        user_transaction_amount=100.0 + (i * 50.0),
                        user_purchase_frequency=1 + (i % 5),
                        logistics_delay_reason=DelayReason.NONE if i % 2 == 0 else DelayReason.TRAFFIC,
                        asset_utilization=70.0 + (i * 3.0),
                        demand_forecast=200.0 + (i * 20.0),
                        logistics_delay=i % 3 == 1,  # Every third shipment is delayed
                        timestamp=datetime.utcnow(),
                    )
                    sample_shipments.append(shipment)
                
                # Bulk insert
                db.bulk_save_objects(sample_shipments)
                db.commit()
                
                logger.info(f"Successfully loaded {len(sample_shipments)} sample shipments")
                return len(sample_shipments)
                
        except Exception as e:
            logger.error(f"Error loading sample data: {e}")
            raise


# Global service instance
data_service = DataService()
