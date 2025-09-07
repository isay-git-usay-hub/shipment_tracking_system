"""
Modern FastAPI application for Maersk Shipment AI System
"""
import logging
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
from sqlalchemy.orm import Session

from config.settings import settings
from core.database import get_db, init_database
from core.schemas import (
    ShipmentResponse, 
    ShipmentSummary, 
    ShipmentCreate,
    ShipmentUpdate,
    ShipmentFilter,
)
from services.shipment_service import shipment_service
from services.data_service_simple import data_service
from services.real_data_loader import real_data_loader

# Import ML service (after logger is configured)
ML_SERVICE_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import ML service now that logger is configured
try:
    import sys
    import os
    ml_path = os.path.join(os.path.dirname(__file__), '..', 'ml')
    sys.path.insert(0, ml_path)
    from services.ml_service import get_ml_service
    ML_SERVICE_AVAILABLE = True
    logger.info("✅ ML service available")
except ImportError as e:
    logger.warning(f"ML service not available: {e}")
    ML_SERVICE_AVAILABLE = False
except Exception as e:
    logger.error(f"Error importing ML service: {e}")
    ML_SERVICE_AVAILABLE = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting Maersk Shipment AI System...")
    
    try:
        # Initialize database
        init_database()
        logger.info("✅ Database initialized")
        
        # Load initial data if needed
        try:
            summary = data_service.get_dataset_summary()
            logger.info(f"📊 Dataset contains {summary['total_records']} records")
        except Exception as e:
            logger.warning(f"Could not load dataset summary: {e}")
        
        logger.info("🎯 Application startup completed")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down application...")


# Create FastAPI app
app = FastAPI(
    title="Maersk Shipment AI System",
    description="""
    🚢 **Modern AI/ML Supply Chain Solution**
    
    A comprehensive shipment tracking and delay prediction system featuring:
    
    - **Real-time shipment monitoring** 📍
    - **AI-powered delay predictions** 🤖
    - **Interactive analytics dashboard** 📊
    - **RESTful API for integrations** 🔗
    
    Built with FastAPI, SQLAlchemy, and modern ML techniques.
    """,
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
try:
    from api.routers.analytics import router as analytics_router, reporting_router
    app.include_router(analytics_router, prefix="/analytics", tags=["Advanced Analytics"])
    app.include_router(reporting_router, prefix="/reports", tags=["Advanced Reports"])
    logger.info("✅ Analytics and Reporting routers included")
except ImportError as e:
    logger.warning(f"Could not import analytics routers: {e}")
except Exception as e:
    logger.error(f"Error including analytics routers: {e}")


# Health Check
@app.get("/health", tags=["Health"])
async def health_check():
    """System health check"""
    return {
        "status": "healthy",
        "service": "Maersk Shipment AI System",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }


# Root endpoint
@app.get("/", tags=["Info"])
async def root():
    """API information"""
    return {
        "message": "🚢 Maersk Shipment AI System API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "timestamp": datetime.utcnow().isoformat(),
        "features": [
            "Real-time shipment tracking",
            "AI-powered delay predictions", 
            "Analytics and reporting",
            "RESTful API"
        ]
    }


# Shipment Endpoints
@app.post("/shipments", response_model=ShipmentResponse, tags=["Shipments"])
async def create_shipment(
    shipment_data: ShipmentCreate,
    db: Session = Depends(get_db)
):
    """Create a new shipment"""
    try:
        shipment = shipment_service.create_shipment(db, shipment_data)
        return shipment
    except Exception as e:
        logger.error(f"Error creating shipment: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/shipments", response_model=List[ShipmentSummary], tags=["Shipments"])
async def get_shipments(
    asset_id: Optional[str] = Query(None, description="Filter by asset ID"),
    shipment_status: Optional[str] = Query(None, description="Filter by status"),
    logistics_delay: Optional[bool] = Query(None, description="Filter by delay status"),
    limit: int = Query(100, le=1000, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Number of records to skip"),
    db: Session = Depends(get_db)
):
    """Get shipments with optional filtering"""
    try:
        # Build filter object
        filters = ShipmentFilter(
            asset_id=asset_id,
            shipment_status=shipment_status,
            logistics_delay=logistics_delay,
            limit=limit,
            offset=offset
        )
        
        shipments = shipment_service.get_shipments(db, filters, skip=offset, limit=limit)
        
        # Convert to summary format
        return [
            {
                "id": s.id,
                "asset_id": s.asset_id,
                "shipment_status": s.shipment_status.value,
                "latitude": s.latitude,
                "longitude": s.longitude,
                "logistics_delay": s.logistics_delay,
                "delay_probability": s.delay_probability,
                "timestamp": s.timestamp,
            }
            for s in shipments
        ]
        
    except Exception as e:
        logger.error(f"Error getting shipments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/shipments/{shipment_id}", response_model=ShipmentResponse, tags=["Shipments"])
async def get_shipment(
    shipment_id: int,
    db: Session = Depends(get_db)
):
    """Get a specific shipment by ID"""
    shipment = shipment_service.get_shipment(db, shipment_id)
    if not shipment:
        raise HTTPException(status_code=404, detail="Shipment not found")
    
    # Convert to response format
    return {
        **shipment.to_dict(),
        "shipment_status": shipment.shipment_status.value,
        "traffic_status": shipment.traffic_status.value if shipment.traffic_status else None,
        "logistics_delay_reason": shipment.logistics_delay_reason.value if shipment.logistics_delay_reason else None,
        "delay_probability": shipment.delay_probability,
        "estimated_delay_hours": shipment.estimated_delay_hours,
    }


@app.put("/shipments/{shipment_id}", response_model=ShipmentResponse, tags=["Shipments"])
async def update_shipment(
    shipment_id: int,
    shipment_update: ShipmentUpdate,
    db: Session = Depends(get_db)
):
    """Update a shipment"""
    try:
        shipment = shipment_service.update_shipment(db, shipment_id, shipment_update)
        if not shipment:
            raise HTTPException(status_code=404, detail="Shipment not found")
        
        return {
            **shipment.to_dict(),
            "shipment_status": shipment.shipment_status.value,
            "traffic_status": shipment.traffic_status.value if shipment.traffic_status else None,
            "logistics_delay_reason": shipment.logistics_delay_reason.value if shipment.logistics_delay_reason else None,
            "delay_probability": shipment.delay_probability,
            "estimated_delay_hours": shipment.estimated_delay_hours,
        }
        
    except Exception as e:
        logger.error(f"Error updating shipment {shipment_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/shipments/{shipment_id}", tags=["Shipments"])
async def delete_shipment(
    shipment_id: int,
    db: Session = Depends(get_db)
):
    """Delete a shipment"""
    success = shipment_service.delete_shipment(db, shipment_id)
    if not success:
        raise HTTPException(status_code=404, detail="Shipment not found")
    
    return {"message": "Shipment deleted successfully"}


# Analytics Endpoints
@app.get("/analytics/overview", tags=["Analytics"])
async def get_analytics_overview(db: Session = Depends(get_db)):
    """Get analytics overview"""
    try:
        stats = shipment_service.get_shipment_statistics(db)
        return {
            "title": "Shipment Analytics Overview",
            **stats
        }
    except Exception as e:
        logger.error(f"Error getting analytics overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/delayed", tags=["Analytics"])
async def get_delayed_shipments(
    hours_back: int = Query(24, ge=1, le=168, description="Hours to look back"),
    db: Session = Depends(get_db)
):
    """Get currently delayed shipments"""
    try:
        delayed_shipments = shipment_service.get_delayed_shipments(db, hours_back)
        return {
            "total_delayed": len(delayed_shipments),
            "hours_back": hours_back,
            "shipments": [
                {
                    "id": s.id,
                    "asset_id": s.asset_id,
                    "status": s.shipment_status.value,
                    "delay_reason": s.logistics_delay_reason.value if s.logistics_delay_reason else None,
                    "waiting_time_minutes": s.waiting_time,
                    "estimated_delay_hours": s.estimated_delay_hours,
                    "timestamp": s.timestamp.isoformat(),
                }
                for s in delayed_shipments
            ]
        }
    except Exception as e:
        logger.error(f"Error getting delayed shipments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Data Management Endpoints
@app.get("/data/summary", tags=["Data"])
async def get_data_summary():
    """Get dataset summary"""
    try:
        return data_service.get_dataset_summary()
    except Exception as e:
        logger.error(f"Error getting data summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/data/load", tags=["Data"])
async def load_data(
    replace_existing: bool = Query(False, description="Replace existing data"),
    batch_size: int = Query(1000, ge=100, le=5000, description="Batch size for loading")
):
    """Load dataset into database"""
    try:
        loaded_count = data_service.load_data_to_database(
            batch_size=batch_size,
            replace_existing=replace_existing
        )
        
        return {
            "message": "Data loaded successfully",
            "records_loaded": loaded_count,
            "replace_existing": replace_existing,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/validate", tags=["Data"])
async def validate_data(db: Session = Depends(get_db)):
    """Validate database data integrity"""
    try:
        return data_service.validate_database_data(db)
    except Exception as e:
        logger.error(f"Error validating data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/real-summary", tags=["Data"])
async def get_real_data_summary():
    """Get real dataset summary from CSV file"""
    try:
        return real_data_loader.get_real_dataset_summary()
    except Exception as e:
        logger.error(f"Error getting real data summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/data/load-real", tags=["Data"])
async def load_real_data(
    replace_existing: bool = Query(False, description="Replace existing data"),
    batch_size: int = Query(500, ge=100, le=2000, description="Batch size for loading")
):
    """Load real dataset from CSV into database"""
    try:
        loaded_count = real_data_loader.load_real_data_to_database(
            batch_size=batch_size,
            replace_existing=replace_existing
        )
        
        return {
            "message": "Real dataset loaded successfully",
            "records_loaded": loaded_count,
            "replace_existing": replace_existing,
            "source": "smart_logistics_dataset.csv",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error loading real data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ML/AI Prediction Endpoints
from pydantic import BaseModel
from typing import Union

@app.get("/ml/status", tags=["Machine Learning"])
async def get_ml_status():
    """Get ML model status and performance metrics"""
    try:
        import sys
        import os
        ml_path = os.path.join(os.path.dirname(__file__), '..', 'ml')
        sys.path.insert(0, ml_path)
        from delay_predictor import get_delay_predictor
        
        predictor = get_delay_predictor()
        
        return {
            "model_available": predictor.is_trained,
            "model_type": "RandomForestClassifier",
            "features": len(predictor.feature_columns) if predictor.is_trained else 0,
            "feature_importance": predictor.get_feature_importance() if predictor.is_trained else {},
            "trained": predictor.is_trained,
            "message": "ML model trained and ready" if predictor.is_trained else "Model not trained"
        }
    except Exception as e:
        return {"model_available": False, "error": str(e)}

@app.post("/ml/predict", tags=["Machine Learning"])
async def predict_delays(
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Get delay predictions for recent shipments using ML model"""
    try:
        import sys
        import os
        import pandas as pd
        ml_path = os.path.join(os.path.dirname(__file__), '..', 'ml')
        sys.path.insert(0, ml_path)
        from delay_predictor import get_delay_predictor
        
        # Get recent shipments
        shipments = shipment_service.get_shipments(db, filters=None, skip=0, limit=limit)
        
        if not shipments:
            return {"error": "No shipments found"}
        
        # Prepare data for prediction
        data = []
        for s in shipments:
            data.append({
                'Asset_ID': s.asset_id,
                'Latitude': s.latitude,
                'Longitude': s.longitude,
                'Inventory_Level': s.inventory_level,
                'Shipment_Status': s.shipment_status.value,
                'Temperature': s.temperature,
                'Humidity': s.humidity,
                'Traffic_Status': s.traffic_status.value if s.traffic_status else 'Clear',
                'Waiting_Time': s.waiting_time,
                'User_Transaction_Amount': s.user_transaction_amount or 0,
                'User_Purchase_Frequency': s.user_purchase_frequency or 0,
                'Logistics_Delay_Reason': s.logistics_delay_reason.value if s.logistics_delay_reason else 'None',
                'Asset_Utilization': s.asset_utilization or 0,
                'Demand_Forecast': s.demand_forecast or 0,
                'Timestamp': s.timestamp,
                'Logistics_Delay': s.logistics_delay  # Actual value for comparison
            })
        
        df = pd.DataFrame(data)
        
        # Get predictions
        predictor = get_delay_predictor()
        predictions, probabilities = predictor.predict(df)
        
        # Format results
        results = []
        for i, s in enumerate(shipments):
            results.append({
                "shipment_id": s.id,
                "asset_id": s.asset_id,
                "actual_delay": bool(s.logistics_delay),
                "predicted_delay": bool(predictions[i]),
                "delay_probability": float(probabilities[i]),
                "correct_prediction": bool(predictions[i]) == bool(s.logistics_delay)
            })
        
        # Calculate accuracy
        correct = sum(1 for r in results if r['correct_prediction'])
        accuracy = correct / len(results) if results else 0
        
        return {
            "model_type": "RandomForestClassifier",
            "predictions": results,
            "accuracy": accuracy,
            "total_predictions": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error in ML prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if ML_SERVICE_AVAILABLE:
    pass  # ML service endpoints were already added above
    
    class DelayPredictionRequest(BaseModel):
        asset_id: str
        latitude: float
        longitude: float
        distance_traveled: float
        fuel_efficiency: float
        waiting_time: int
        status: str
        traffic_conditions: str
        weather_conditions: str
        delay_reason: Optional[str] = None
        timestamp: Optional[str] = None
    
    class BatchPredictionRequest(BaseModel):
        shipments: List[dict]
    
    @app.post("/ml/predict", tags=["ML/AI"])
    async def predict_delay(prediction_request: DelayPredictionRequest):
        """Predict delay probability for a single shipment"""
        try:
            ml_service = get_ml_service()
            result = await ml_service.predict_delay(prediction_request.dict())
            
            if not result['success']:
                raise HTTPException(status_code=400, detail=result['error'])
            
            return result['prediction']
            
        except Exception as e:
            logger.error(f"Error in delay prediction: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/ml/predict-batch", tags=["ML/AI"])
    async def predict_delays_batch(batch_request: BatchPredictionRequest):
        """Predict delays for multiple shipments"""
        try:
            ml_service = get_ml_service()
            result = await ml_service.predict_batch(batch_request.shipments)
            
            if not result['success']:
                raise HTTPException(status_code=400, detail=result['error'])
            
            return {
                "predictions": result['predictions'],
                "summary": result['summary']
            }
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/ml/model/info", tags=["ML/AI"])
    async def get_model_info():
        """Get information about the active ML model"""
        try:
            ml_service = get_ml_service()
            return ml_service.get_model_info()
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/ml/models", tags=["ML/AI"])
    async def list_saved_models():
        """List all saved ML models"""
        try:
            ml_service = get_ml_service()
            return {"models": ml_service.list_saved_models()}
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/ml/train", tags=["ML/AI"])
    async def train_models(
        data_source: str = Query("real", description="Data source: 'real' or 'sample'"),
        model_types: Optional[List[str]] = Query(None, description="Specific model types to train")
    ):
        """Train ML models"""
        try:
            ml_service = get_ml_service()
            
            # Determine data path
            if data_source == "real":
                data_path = "data/smart_logistics_dataset.csv"
            else:
                # For sample data, we'd need to export from database first
                raise HTTPException(status_code=400, detail="Sample data training not yet implemented")
            
            result = await ml_service.train_models(data_path, model_types)
            
            if not result['success']:
                raise HTTPException(status_code=400, detail=result['error'])
            
            return {
                "message": "Model training completed",
                "best_model": result['best_model'],
                "filename": result['filename'],
                "training_results": {name: res.get('test_metrics', {}) for name, res in result['results'].items()}
            }
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/ml/feature-importance", tags=["ML/AI"])
    async def get_feature_importance():
        """Get feature importance from the active model"""
        try:
            ml_service = get_ml_service()
            result = ml_service.get_feature_importance()
            
            if not result['success']:
                raise HTTPException(status_code=400, detail=result['error'])
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/ml/evaluate", tags=["ML/AI"])
    async def evaluate_model(
        data_source: str = Query("real", description="Data source for evaluation")
    ):
        """Evaluate the active model on test data"""
        try:
            ml_service = get_ml_service()
            
            # Determine data path
            if data_source == "real":
                data_path = "data/smart_logistics_dataset.csv"
            else:
                raise HTTPException(status_code=400, detail="Sample data evaluation not yet implemented")
            
            result = await ml_service.evaluate_model(data_path)
            
            if not result['success']:
                raise HTTPException(status_code=400, detail=result['error'])
            
            return {
                "message": "Model evaluation completed",
                "metrics": result['metrics'],
                "evaluation_size": result['evaluation_size'],
                "model_name": result['model_name']
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise HTTPException(status_code=500, detail=str(e))
else:
    # Placeholder endpoints when ML service is not available
    @app.get("/ml/status", tags=["ML/AI"])
    async def ml_status():
        """ML service status"""
        return {
            "available": False,
            "message": "ML service not available. Check dependencies and configuration."
        }


# Notification System API Endpoints
try:
    import sys
    import os
    notifications_path = os.path.join(os.path.dirname(__file__), '..', 'notifications')
    sys.path.insert(0, notifications_path)
    from notification_service import get_notification_service
    from alert_monitor import get_alert_monitor
    NOTIFICATIONS_AVAILABLE = True
    logger.info("✅ Notification system available")
except ImportError as e:
    logger.warning(f"Notification system not available: {e}")
    NOTIFICATIONS_AVAILABLE = False
except Exception as e:
    logger.error(f"Error importing notification system: {e}")
    NOTIFICATIONS_AVAILABLE = False

if NOTIFICATIONS_AVAILABLE:
    from pydantic import BaseModel
    from typing import Union
    
    class NotificationRecipientCreate(BaseModel):
        name: str
        email: Optional[str] = None
        phone: Optional[str] = None
        webhook_url: Optional[str] = None
        slack_channel: Optional[str] = None
        active: bool = True
    
    class TestNotificationRequest(BaseModel):
        notification_type: str
        test_data: dict
        recipients: Optional[List[str]] = None
    
    @app.get("/notifications/status", tags=["Notifications"])
    async def get_notification_status():
        """Get notification system status"""
        try:
            notification_service = get_notification_service()
            alert_monitor = get_alert_monitor()
            
            return {
                "notification_service_active": True,
                "alert_monitoring_active": alert_monitor.monitoring_active,
                "notification_stats": notification_service.get_notification_stats(),
                "monitoring_status": alert_monitor.get_monitoring_status()
            }
        except Exception as e:
            logger.error(f"Error getting notification status: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/notifications/recipients", tags=["Notifications"])
    async def add_notification_recipient(
        recipient_id: str,
        recipient: NotificationRecipientCreate
    ):
        """Add a notification recipient"""
        try:
            notification_service = get_notification_service()
            
            from notification_service import NotificationRecipient
            new_recipient = NotificationRecipient(
                name=recipient.name,
                email=recipient.email,
                phone=recipient.phone,
                webhook_url=recipient.webhook_url,
                slack_channel=recipient.slack_channel,
                active=recipient.active
            )
            
            notification_service.add_recipient(recipient_id, new_recipient)
            
            return {
                "message": f"Recipient {recipient.name} added successfully",
                "recipient_id": recipient_id
            }
        except Exception as e:
            logger.error(f"Error adding recipient: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/notifications/history", tags=["Notifications"])
    async def get_notification_history(
        limit: int = Query(50, le=200, description="Number of notifications to retrieve")
    ):
        """Get notification history"""
        try:
            notification_service = get_notification_service()
            history = notification_service.get_notification_history(limit)
            
            return {
                "total_returned": len(history),
                "notifications": history
            }
        except Exception as e:
            logger.error(f"Error getting notification history: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/notifications/rules", tags=["Notifications"])
    async def get_alert_rules():
        """Get all alert rules and their configurations"""
        try:
            alert_monitor = get_alert_monitor()
            return {
                "rules": alert_monitor.get_alert_rules_summary()
            }
        except Exception as e:
            logger.error(f"Error getting alert rules: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/notifications/rules/{rule_name}/toggle", tags=["Notifications"])
    async def toggle_alert_rule(
        rule_name: str,
        enabled: bool = Query(True, description="Enable or disable the rule")
    ):
        """Enable or disable an alert rule"""
        try:
            alert_monitor = get_alert_monitor()
            alert_monitor.enable_rule(rule_name, enabled)
            
            return {
                "message": f"Rule '{rule_name}' {'enabled' if enabled else 'disabled'}",
                "rule_name": rule_name,
                "enabled": enabled
            }
        except Exception as e:
            logger.error(f"Error toggling rule: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/notifications/monitoring/start", tags=["Notifications"])
    async def start_monitoring():
        """Start the alert monitoring system"""
        try:
            alert_monitor = get_alert_monitor()
            alert_monitor.start_monitoring()
            
            return {
                "message": "Alert monitoring started",
                "monitoring_active": True
            }
        except Exception as e:
            logger.error(f"Error starting monitoring: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/notifications/monitoring/stop", tags=["Notifications"])
    async def stop_monitoring():
        """Stop the alert monitoring system"""
        try:
            alert_monitor = get_alert_monitor()
            alert_monitor.stop_monitoring()
            
            return {
                "message": "Alert monitoring stopped",
                "monitoring_active": False
            }
        except Exception as e:
            logger.error(f"Error stopping monitoring: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/notifications/test", tags=["Notifications"])
    async def send_test_notification(test_request: TestNotificationRequest):
        """Send a test notification"""
        try:
            notification_service = get_notification_service()
            
            # Create test notification based on type
            if test_request.notification_type == "delay_prediction":
                notification_id = await notification_service.create_delay_prediction_alert(
                    test_request.test_data
                )
            elif test_request.notification_type == "status_change":
                notification_id = await notification_service.create_status_change_alert(
                    test_request.test_data
                )
            elif test_request.notification_type == "high_risk":
                notification_id = await notification_service.create_high_risk_alert(
                    test_request.test_data
                )
            elif test_request.notification_type == "system_alert":
                notification_id = await notification_service.create_system_alert(
                    test_request.test_data
                )
            else:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unknown notification type: {test_request.notification_type}"
                )
            
            return {
                "message": "Test notification queued",
                "notification_id": notification_id,
                "type": test_request.notification_type
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error sending test notification: {e}")
            raise HTTPException(status_code=500, detail=str(e))
else:
    # Placeholder endpoints when notification system is not available
    @app.get("/notifications/status", tags=["Notifications"])
    async def notification_status():
        """Notification system status"""
        return {
            "available": False,
            "message": "Notification system not available. Check dependencies and configuration."
        }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main_new:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level=settings.LOG_LEVEL.lower()
    )
