"""
Load the real smart logistics dataset into the Maersk system
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
import random

from models.database.connection import SessionLocal, create_tables_sync
from models.database.models import (
    Customer, Shipment, DelayPrediction, CommunicationLog,
    ShipmentStatus, DelayReason, CommunicationType
)


def load_smart_logistics_data():
    """Load the smart logistics dataset and convert it to our system format"""
    print("Loading smart logistics dataset...")
    
    # Read the dataset
    df = pd.read_csv('data/smart_logistics_dataset.csv')
    print(f"Loaded {len(df)} records from dataset")
    
    # Create database session
    db = SessionLocal()
    
    try:
        # Clear existing data
        print("Clearing existing data...")
        db.query(CommunicationLog).delete()
        db.query(DelayPrediction).delete()
        db.query(Shipment).delete()
        db.query(Customer).delete()
        db.commit()
        
        # Create customers based on unique assets
        create_customers_from_assets(db, df)
        
        # Create shipments from the dataset
        create_shipments_from_data(db, df)
        
        # Create predictions based on actual delays
        create_predictions_from_data(db, df)
        
        # Create communication logs
        create_communications_from_data(db, df)
        
        print("Successfully loaded real dataset into system!")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def create_customers_from_assets(db: Session, df: pd.DataFrame):
    """Create customers based on unique truck assets"""
    print("Creating customers from assets...")
    
    # Get unique assets
    unique_assets = df['Asset_ID'].unique()
    
    companies = [
        "Global Logistics Corp", "Maritime Transport Ltd", "Ocean Freight Solutions",
        "Continental Shipping", "Express Cargo Services", "International Trade Co",
        "Worldwide Logistics", "Premier Transport", "Cargo Express Inc", "Supply Chain Masters"
    ]
    
    customers = []
    for i, asset in enumerate(unique_assets):
        customer = Customer(
            name=f"Fleet Owner {i+1}",
            email=f"fleet{i+1}@{companies[i % len(companies)].lower().replace(' ', '')}.com",
            phone=f"+1-555-{random.randint(1000, 9999)}",
            company=companies[i % len(companies)],
            preferred_communication=random.choice(list(CommunicationType)),
            language_preference="en",
            timezone="UTC"
        )
        customers.append(customer)
    
    db.add_all(customers)
    db.commit()
    print(f"Created {len(customers)} customers")


def create_shipments_from_data(db: Session, df: pd.DataFrame):
    """Create shipments from the dataset"""
    print("Creating shipments from dataset...")
    
    customers = db.query(Customer).all()
    customer_map = {f"Truck_{i+1}": customers[i % len(customers)] for i in range(len(customers))}
    
    # Port mapping based on coordinates (simplified)
    def get_port_from_coordinates(lat, lng):
        ports = [
            "Shanghai", "Singapore", "Rotterdam", "Los Angeles", "Hamburg",
            "Antwerp", "Hong Kong", "Dubai", "New York", "Bremen"
        ]
        # Simple hash-based port assignment for consistency
        port_index = abs(hash(f"{lat:.1f},{lng:.1f}")) % len(ports)
        return ports[port_index]
    
    # Status mapping
    status_map = {
        'Delivered': ShipmentStatus.DELIVERED,
        'Delayed': ShipmentStatus.DELAYED,
        'In Transit': ShipmentStatus.IN_TRANSIT,
        'Pending': ShipmentStatus.PENDING
    }
    
    shipments = []
    for idx, row in df.iterrows():
        if idx >= 500:  # Limit to first 500 records for performance
            break
            
        asset_id = row['Asset_ID']
        customer = customer_map.get(asset_id, customers[0])
        
        # Generate origin and destination ports from coordinates
        origin_port = get_port_from_coordinates(row['Latitude'], row['Longitude'])
        # Vary destination port
        all_ports = ["Shanghai", "Singapore", "Rotterdam", "Los Angeles", "Hamburg"]
        destination_port = random.choice([p for p in all_ports if p != origin_port])
        
        # Parse timestamp
        timestamp = pd.to_datetime(row['Timestamp'])
        
        # Generate realistic departure and arrival times
        departure_time = timestamp - timedelta(days=random.randint(1, 10))
        transit_days = random.randint(7, 21)
        scheduled_arrival = departure_time + timedelta(days=transit_days)
        
        # Calculate actual arrival based on delay
        actual_arrival = None
        if row['Shipment_Status'] == 'Delivered':
            delay_hours = row['Waiting_Time'] if row['Logistics_Delay'] else 0
            actual_arrival = scheduled_arrival + timedelta(hours=delay_hours)
        
        shipment = Shipment(
            shipment_id=f"MSK{1000 + idx}",
            customer_id=customer.id,
            origin_port=origin_port,
            destination_port=destination_port,
            route=f"{origin_port}-{destination_port}",
            container_type=random.choice(["20GP", "40GP", "40HC", "45HC"]),
            cargo_weight=row['Inventory_Level'] * 10,  # Scale up inventory to kg
            cargo_value=random.randint(50000, 500000),
            cargo_type=random.choice(["electronics", "textiles", "machinery", "chemicals"]),
            scheduled_departure=departure_time,
            scheduled_arrival=scheduled_arrival,
            actual_departure=departure_time + timedelta(hours=random.randint(0, 4)),
            estimated_arrival=scheduled_arrival + timedelta(hours=row['Waiting_Time']),
            actual_arrival=actual_arrival,
            status=status_map.get(row['Shipment_Status'], ShipmentStatus.IN_TRANSIT),
            is_priority=random.choice([True, False])
        )
        shipments.append(shipment)
    
    db.add_all(shipments)
    db.commit()
    print(f"Created {len(shipments)} shipments")


def create_predictions_from_data(db: Session, df: pd.DataFrame):
    """Create delay predictions based on the dataset"""
    print("Creating predictions from dataset...")
    
    shipments = db.query(Shipment).all()
    
    predictions = []
    for i, shipment in enumerate(shipments[:300]):  # Limit to first 300
        if i >= len(df):
            break
            
        row = df.iloc[i]
        
        # Use actual data to create realistic predictions
        actual_delay = row['Waiting_Time'] if row['Logistics_Delay'] else 0
        predicted_delay = actual_delay + random.gauss(0, 3)  # Add some noise
        predicted_delay = max(0, predicted_delay)
        
        delay_probability = min(1.0, predicted_delay / 48.0)
        confidence_score = 0.7 + random.random() * 0.25
        
        # Calculate accuracy
        accuracy = None
        if actual_delay > 0:
            accuracy = 1 - abs(predicted_delay - actual_delay) / max(predicted_delay, actual_delay, 1)
            accuracy = max(0, min(1, accuracy))
        
        prediction = DelayPrediction(
            shipment_id=shipment.id,
            predicted_delay_hours=predicted_delay,
            delay_probability=delay_probability,
            confidence_score=confidence_score,
            model_name="RealData-XGBoost",
            model_version="1.0",
            features_used={
                "temperature": float(row['Temperature']),
                "humidity": float(row['Humidity']),
                "traffic_status": row['Traffic_Status'],
                "asset_utilization": float(row['Asset_Utilization']),
                "demand_forecast": float(row['Demand_Forecast'])
            },
            weather_factor=row['Temperature'] / 30.0,  # Normalize
            port_congestion_factor=row['Asset_Utilization'] / 100.0,
            route_complexity_factor=0.5,
            prediction_date=pd.to_datetime(row['Timestamp']),
            actual_delay_hours=actual_delay if row['Logistics_Delay'] else 0,
            prediction_accuracy=accuracy
        )
        predictions.append(prediction)
    
    db.add_all(predictions)
    db.commit()
    print(f"Created {len(predictions)} predictions")


def create_communications_from_data(db: Session, df: pd.DataFrame):
    """Create communication logs based on delays in dataset"""
    print("Creating communications from dataset...")
    
    shipments = db.query(Shipment).all()
    
    communications = []
    delayed_shipments = [s for s in shipments if s.status == ShipmentStatus.DELAYED]
    
    for i, shipment in enumerate(delayed_shipments[:200]):  # Limit to 200 communications
        if i >= len(df):
            break
            
        row = df.iloc[i]
        comm_type = random.choice(list(CommunicationType))
        
        # Generate realistic communication content
        delay_reason = row.get('Logistics_Delay_Reason', 'Unknown')
        if pd.isna(delay_reason) or delay_reason == 'None' or str(delay_reason) == 'nan':
            delay_reason = 'Operational delays'
        
        if comm_type == CommunicationType.EMAIL:
            subject = f"Shipment Update - {shipment.shipment_id}"
            message = f"Dear {shipment.customer.name},\n\nWe want to update you on your shipment {shipment.shipment_id}. Due to {delay_reason.lower()}, there may be a delay in delivery. We are working to minimize any inconvenience.\n\nBest regards,\nMaersk Team"
            recipient = shipment.customer.email
        else:
            subject = None
            message = f"Maersk Alert: Shipment {shipment.shipment_id} delayed due to {delay_reason.lower()}. New ETA will be provided soon."
            recipient = shipment.customer.phone or "+1-555-0000"
        
        communication = CommunicationLog(
            shipment_id=shipment.id,
            customer_id=shipment.customer_id,
            type=comm_type,
            subject=subject,
            message=message,
            recipient=recipient,
            ai_generated=True,
            model_used="RealData-GPT",
            template_used="delay_notification",
            personalization_data={
                "delay_reason": delay_reason,
                "waiting_time": row['Waiting_Time'],
                "traffic_status": row['Traffic_Status']
            },
            sent_at=pd.to_datetime(row['Timestamp']),
            delivery_status="delivered"
        )
        communications.append(communication)
    
    db.add_all(communications)
    db.commit()
    print(f"Created {len(communications)} communications")


def get_dataset_summary():
    """Get a summary of the dataset"""
    df = pd.read_csv('data/smart_logistics_dataset.csv')
    
    print("=== Dataset Summary ===")
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
    print(f"Unique assets: {df['Asset_ID'].nunique()}")
    print("\nShipment Status Distribution:")
    print(df['Shipment_Status'].value_counts())
    print("\nDelay Distribution:")
    print(df['Logistics_Delay'].value_counts())
    print("\nDelay Reasons:")
    print(df['Logistics_Delay_Reason'].value_counts())
    
    return df


if __name__ == "__main__":
    # Show dataset summary first
    get_dataset_summary()
    
    # Create tables
    create_tables_sync()
    
    # Load the data
    load_smart_logistics_data()
    
    print("\n=== Data Loading Complete! ===")
    print("Your Maersk AI System now contains real logistics data:")
    print("- Shipment records based on actual truck movements")  
    print("- Delay predictions using real delay patterns")
    print("- Communication logs for delayed shipments")
    print("- Environmental and operational data (temperature, humidity, traffic)")
