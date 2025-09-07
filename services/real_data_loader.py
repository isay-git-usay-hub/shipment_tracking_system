"""
Real data loader for smart_logistics_dataset.csv
"""
import logging
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from sqlalchemy.orm import Session

from core.models import Shipment, ShipmentStatus, TrafficStatus, DelayReason
from core.database import get_db_session

logger = logging.getLogger(__name__)


class RealDataLoader:
    """Service for loading the real smart logistics dataset"""
    
    def __init__(self):
        self.data_path = Path("data")
        self.dataset_file = self.data_path / "smart_logistics_dataset.csv"
    
    def _parse_datetime(self, datetime_str: str) -> datetime:
        """Parse datetime from string"""
        try:
            return datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            # Fallback parsing
            try:
                return datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
            except:
                return datetime.utcnow()
    
    def _safe_float(self, value: str) -> Optional[float]:
        """Safely convert string to float"""
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _safe_int(self, value: str) -> Optional[int]:
        """Safely convert string to int"""
        try:
            return int(float(value))  # Handle cases like "123.0"
        except (ValueError, TypeError):
            return None
    
    def _map_shipment_status(self, status: str) -> ShipmentStatus:
        """Map dataset status to model enum"""
        status_mapping = {
            'In Transit': ShipmentStatus.IN_TRANSIT,
            'Delivered': ShipmentStatus.DELIVERED,
            'Delayed': ShipmentStatus.DELAYED,
        }
        return status_mapping.get(status, ShipmentStatus.IN_TRANSIT)
    
    def _map_traffic_status(self, status: str) -> TrafficStatus:
        """Map dataset traffic status to model enum"""
        if not status or status == '':
            return TrafficStatus.CLEAR
            
        traffic_mapping = {
            'Clear': TrafficStatus.CLEAR,
            'Heavy': TrafficStatus.HEAVY,
            'Detour': TrafficStatus.DETOUR,
        }
        return traffic_mapping.get(status, TrafficStatus.CLEAR)
    
    def _map_delay_reason(self, reason: str) -> DelayReason:
        """Map dataset delay reason to model enum"""
        if not reason or reason == 'None' or reason == '':
            return DelayReason.NONE
            
        reason_mapping = {
            'Weather': DelayReason.WEATHER,
            'Traffic': DelayReason.TRAFFIC,
            'Mechanical Failure': DelayReason.MECHANICAL_FAILURE,
            'None': DelayReason.NONE,
        }
        return reason_mapping.get(reason, DelayReason.OTHER)
    
    def load_real_dataset(self) -> List[Dict[str, Any]]:
        """Load the real smart logistics dataset"""
        try:
            if not self.dataset_file.exists():
                raise FileNotFoundError(f"Dataset file not found: {self.dataset_file}")
            
            logger.info(f"Loading real dataset from {self.dataset_file}")
            
            records = []
            with open(self.dataset_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    # Clean and validate each record
                    cleaned_record = self._clean_record(row)
                    if cleaned_record:
                        records.append(cleaned_record)
            
            logger.info(f"Loaded {len(records)} valid records from dataset")
            return records
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def _clean_record(self, row: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Clean and validate a single record"""
        try:
            # Parse and validate required fields
            latitude = self._safe_float(row.get('Latitude'))
            longitude = self._safe_float(row.get('Longitude'))
            inventory_level = self._safe_int(row.get('Inventory_Level'))
            waiting_time = self._safe_int(row.get('Waiting_Time'))
            
            # Validate required fields
            if latitude is None or longitude is None or inventory_level is None or waiting_time is None:
                return None
            
            # Validate coordinate ranges
            if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
                return None
            
            # Validate other numeric ranges
            if inventory_level < 0 or waiting_time < 0:
                return None
            
            return {
                'timestamp': self._parse_datetime(row.get('Timestamp', '')),
                'asset_id': row.get('Asset_ID', '').strip(),
                'latitude': latitude,
                'longitude': longitude,
                'inventory_level': inventory_level,
                'shipment_status': self._map_shipment_status(row.get('Shipment_Status', '')),
                'temperature': self._safe_float(row.get('Temperature')),
                'humidity': self._safe_float(row.get('Humidity')),
                'traffic_status': self._map_traffic_status(row.get('Traffic_Status')),
                'waiting_time': waiting_time,
                'user_transaction_amount': self._safe_float(row.get('User_Transaction_Amount')),
                'user_purchase_frequency': self._safe_int(row.get('User_Purchase_Frequency')),
                'logistics_delay_reason': self._map_delay_reason(row.get('Logistics_Delay_Reason')),
                'asset_utilization': self._safe_float(row.get('Asset_Utilization')),
                'demand_forecast': self._safe_float(row.get('Demand_Forecast')),
                'logistics_delay': row.get('Logistics_Delay') == '1' or row.get('Logistics_Delay') == 'True',
            }
            
        except Exception as e:
            logger.warning(f"Error processing record: {e}")
            return None
    
    def load_real_data_to_database(self, batch_size: int = 1000, replace_existing: bool = False) -> int:
        """Load real dataset into database"""
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
                
                # Load and process real dataset
                records = self.load_real_dataset()
                total_records = len(records)
                loaded_count = 0
                
                if total_records == 0:
                    logger.warning("No valid records found in dataset")
                    return 0
                
                logger.info(f"Loading {total_records} shipments to database...")
                
                # Process in batches
                for i in range(0, total_records, batch_size):
                    batch_records = records[i:i+batch_size]
                    batch_shipments = []
                    
                    for record in batch_records:
                        try:
                            # Create shipment object
                            shipment = Shipment(**record)
                            batch_shipments.append(shipment)
                            
                        except Exception as e:
                            logger.warning(f"Skipping invalid record: {e}")
                            continue
                    
                    # Bulk insert batch
                    if batch_shipments:
                        db.bulk_save_objects(batch_shipments)
                        db.commit()
                        loaded_count += len(batch_shipments)
                        
                        logger.info(f"Loaded batch {i//batch_size + 1}: {loaded_count}/{total_records} records")
                
                logger.info(f"Successfully loaded {loaded_count} shipments from real dataset")
                return loaded_count
                
        except Exception as e:
            logger.error(f"Error loading real data to database: {e}")
            raise
    
    def get_real_dataset_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the real dataset"""
        try:
            if not self.dataset_file.exists():
                return {
                    "status": "Dataset file not found",
                    "total_records": 0,
                    "file_path": str(self.dataset_file)
                }
            
            records = self.load_real_dataset()
            
            if not records:
                return {
                    "status": "No valid records found",
                    "total_records": 0,
                    "file_path": str(self.dataset_file)
                }
            
            # Calculate statistics
            total_delayed = sum(1 for r in records if r['logistics_delay'])
            delay_rate = (total_delayed / len(records) * 100) if records else 0
            
            # Get date range
            timestamps = [r['timestamp'] for r in records]
            min_date = min(timestamps)
            max_date = max(timestamps)
            
            # Asset distribution
            assets = set(r['asset_id'] for r in records)
            
            # Status distribution
            status_counts = {}
            for record in records:
                status = record['shipment_status'].value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Delay reason distribution
            delay_reasons = {}
            for record in records:
                if record['logistics_delay']:
                    reason = record['logistics_delay_reason'].value
                    delay_reasons[reason] = delay_reasons.get(reason, 0) + 1
            
            # Waiting time statistics
            waiting_times = [r['waiting_time'] for r in records]
            
            return {
                "status": "Real dataset loaded successfully",
                "total_records": len(records),
                "file_path": str(self.dataset_file),
                "file_size_mb": round(self.dataset_file.stat().st_size / 1024 / 1024, 2),
                "date_range": {
                    "start": min_date.isoformat(),
                    "end": max_date.isoformat()
                },
                "assets": {
                    "total_assets": len(assets),
                    "asset_list": sorted(list(assets))
                },
                "shipment_status": status_counts,
                "delay_analysis": {
                    "total_delayed": total_delayed,
                    "delay_rate": round(delay_rate, 2)
                },
                "delay_reasons": delay_reasons,
                "waiting_time": {
                    "min": min(waiting_times),
                    "max": max(waiting_times),
                    "avg": round(sum(waiting_times) / len(waiting_times), 2)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating real dataset summary: {e}")
            return {
                "status": "Error loading real dataset",
                "error": str(e),
                "total_records": 0
            }


# Global service instance
real_data_loader = RealDataLoader()
