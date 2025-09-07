"""
Data service for loading and processing smart logistics dataset
"""
import logging
import csv
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
from sqlalchemy.orm import Session

from core.models import Shipment, ShipmentStatus, TrafficStatus, DelayReason
from core.database import get_db_session
from config.settings import settings

logger = logging.getLogger(__name__)


class DataService:
    """Service for data loading and processing"""
    
    def __init__(self):
        self.data_path = Path("data")
        self.dataset_file = self.data_path / "smart_logistics_dataset.csv"
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load the smart logistics dataset"""
        try:
            if not self.dataset_file.exists():
                raise FileNotFoundError(f"Dataset file not found: {self.dataset_file}")
            
            logger.info(f"Loading dataset from {self.dataset_file}")
            
            records = []
            with open(self.dataset_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    records.append(row)
            
            logger.info(f"Loaded {len(records)} records from dataset")
            return records
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def _clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the dataset"""
        try:
            # Parse timestamp
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            
            # Handle missing values
            df['Temperature'] = df['Temperature'].fillna(df['Temperature'].mean())
            df['Humidity'] = df['Humidity'].fillna(df['Humidity'].mean())
            df['User_Transaction_Amount'] = df['User_Transaction_Amount'].fillna(0)
            df['User_Purchase_Frequency'] = df['User_Purchase_Frequency'].fillna(1)
            df['Asset_Utilization'] = df['Asset_Utilization'].fillna(df['Asset_Utilization'].mean())
            df['Demand_Forecast'] = df['Demand_Forecast'].fillna(df['Demand_Forecast'].mean())
            
            # Handle 'None' string values in delay reason
            df['Logistics_Delay_Reason'] = df['Logistics_Delay_Reason'].fillna('None')
            df['Traffic_Status'] = df['Traffic_Status'].fillna('Clear')
            
            # Remove any duplicate records
            df = df.drop_duplicates()
            
            # Validate numeric ranges
            df = df[(df['Latitude'] >= -90) & (df['Latitude'] <= 90)]
            df = df[(df['Longitude'] >= -180) & (df['Longitude'] <= 180)]
            df = df[df['Waiting_Time'] >= 0]
            df = df[df['Inventory_Level'] >= 0]
            
            logger.info(f"Cleaned dataset: {len(df)} valid records")
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning dataset: {e}")
            raise
    
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
        traffic_mapping = {
            'Clear': TrafficStatus.CLEAR,
            'Heavy': TrafficStatus.HEAVY,
            'Detour': TrafficStatus.DETOUR,
        }
        return traffic_mapping.get(status, TrafficStatus.CLEAR)
    
    def _map_delay_reason(self, reason: str) -> DelayReason:
        """Map dataset delay reason to model enum"""
        if pd.isna(reason) or reason == 'None' or reason == '':
            return DelayReason.NONE
            
        reason_mapping = {
            'Weather': DelayReason.WEATHER,
            'Traffic': DelayReason.TRAFFIC,
            'Mechanical Failure': DelayReason.MECHANICAL_FAILURE,
            'None': DelayReason.NONE,
        }
        return reason_mapping.get(reason, DelayReason.OTHER)
    
    def load_data_to_database(self, batch_size: int = 1000, replace_existing: bool = False) -> int:
        """Load dataset into database"""
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
                
                # Load and process dataset
                df = self.load_dataset()
                total_records = len(df)
                loaded_count = 0
                
                logger.info(f"Loading {total_records} shipments to database...")
                
                # Process in batches
                for i in range(0, total_records, batch_size):
                    batch_df = df.iloc[i:i+batch_size]
                    batch_shipments = []
                    
                    for _, row in batch_df.iterrows():
                        try:
                            # Create shipment object
                            shipment = Shipment(
                                asset_id=str(row['Asset_ID']),
                                latitude=float(row['Latitude']),
                                longitude=float(row['Longitude']),
                                inventory_level=int(row['Inventory_Level']),
                                shipment_status=self._map_shipment_status(row['Shipment_Status']),
                                temperature=float(row['Temperature']) if pd.notna(row['Temperature']) else None,
                                humidity=float(row['Humidity']) if pd.notna(row['Humidity']) else None,
                                traffic_status=self._map_traffic_status(row['Traffic_Status']),
                                waiting_time=int(row['Waiting_Time']),
                                user_transaction_amount=float(row['User_Transaction_Amount']) if pd.notna(row['User_Transaction_Amount']) else None,
                                user_purchase_frequency=int(row['User_Purchase_Frequency']) if pd.notna(row['User_Purchase_Frequency']) else None,
                                logistics_delay_reason=self._map_delay_reason(row['Logistics_Delay_Reason']),
                                asset_utilization=float(row['Asset_Utilization']) if pd.notna(row['Asset_Utilization']) else None,
                                demand_forecast=float(row['Demand_Forecast']) if pd.notna(row['Demand_Forecast']) else None,
                                logistics_delay=bool(row['Logistics_Delay']),
                                timestamp=row['Timestamp'],
                            )
                            batch_shipments.append(shipment)
                            
                        except Exception as e:
                            logger.warning(f"Skipping invalid record at index {i}: {e}")
                            continue
                    
                    # Bulk insert batch
                    if batch_shipments:
                        db.bulk_save_objects(batch_shipments)
                        db.commit()
                        loaded_count += len(batch_shipments)
                        
                        logger.info(f"Loaded batch {i//batch_size + 1}: {loaded_count}/{total_records} records")
                
                logger.info(f"Successfully loaded {loaded_count} shipments to database")
                return loaded_count
                
        except Exception as e:
            logger.error(f"Error loading data to database: {e}")
            raise
    
    def get_dataset_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the dataset"""
        try:
            df = self.load_dataset()
            
            summary = {
                "total_records": len(df),
                "date_range": {
                    "start": df['Timestamp'].min().isoformat(),
                    "end": df['Timestamp'].max().isoformat()
                },
                "assets": {
                    "total_assets": df['Asset_ID'].nunique(),
                    "asset_list": sorted(df['Asset_ID'].unique().tolist())
                },
                "shipment_status": df['Shipment_Status'].value_counts().to_dict(),
                "traffic_status": df['Traffic_Status'].value_counts().to_dict(),
                "delay_reasons": df['Logistics_Delay_Reason'].value_counts().to_dict(),
                "delays": {
                    "total_delayed": int(df['Logistics_Delay'].sum()),
                    "delay_rate": float(df['Logistics_Delay'].mean() * 100)
                },
                "waiting_time": {
                    "min": float(df['Waiting_Time'].min()),
                    "max": float(df['Waiting_Time'].max()),
                    "mean": float(df['Waiting_Time'].mean()),
                    "median": float(df['Waiting_Time'].median())
                },
                "geographic_bounds": {
                    "latitude_range": [float(df['Latitude'].min()), float(df['Latitude'].max())],
                    "longitude_range": [float(df['Longitude'].min()), float(df['Longitude'].max())]
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating dataset summary: {e}")
            raise
    
    def validate_database_data(self, db: Session) -> Dict[str, Any]:
        """Validate data integrity in database"""
        try:
            total_shipments = db.query(Shipment).filter(Shipment.is_active == True).count()
            delayed_shipments = db.query(Shipment).filter(
                Shipment.is_active == True, 
                Shipment.logistics_delay == True
            ).count()
            
            # Get date range
            date_stats = db.query(
                db.func.min(Shipment.timestamp).label('min_date'),
                db.func.max(Shipment.timestamp).label('max_date')
            ).first()
            
            validation_result = {
                "total_records": total_shipments,
                "delayed_records": delayed_shipments,
                "delay_rate": (delayed_shipments / total_shipments * 100) if total_shipments > 0 else 0,
                "date_range": {
                    "start": date_stats.min_date.isoformat() if date_stats.min_date else None,
                    "end": date_stats.max_date.isoformat() if date_stats.max_date else None
                },
                "data_integrity": "VALID" if total_shipments > 0 else "NO_DATA",
                "validation_timestamp": datetime.utcnow().isoformat()
            }
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating database data: {e}")
            raise


# Global service instance
data_service = DataService()
