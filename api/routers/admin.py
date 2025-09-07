"""
Admin endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Dict, Any

from models.database.connection import get_db
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/system/status")
async def get_system_status(db: Session = Depends(get_db)):
    """Get comprehensive system status"""
    try:
        from models.database.models import (
            Shipment, Customer, DelayPrediction, CommunicationLog, 
            ModelPerformance, ExternalData
        )
        from sqlalchemy import func

        # Database statistics
        db_stats = {
            "shipments": db.query(Shipment).count(),
            "customers": db.query(Customer).count(),
            "predictions": db.query(DelayPrediction).count(),
            "communications": db.query(CommunicationLog).count(),
            "model_performance_records": db.query(ModelPerformance).count(),
            "external_data_records": db.query(ExternalData).count()
        }

        # Recent activity (last 24 hours)
        yesterday = datetime.now() - timedelta(days=1)
        recent_activity = {
            "new_shipments": db.query(Shipment).filter(Shipment.created_at >= yesterday).count(),
            "predictions_generated": db.query(DelayPrediction).filter(DelayPrediction.prediction_date >= yesterday).count(),
            "communications_sent": db.query(CommunicationLog).filter(CommunicationLog.sent_at >= yesterday).count()
        }

        # Model status
        from services.prediction.service import prediction_service
        model_status = {}
        for name, model in prediction_service.models.items():
            model_status[name] = {
                "loaded": model is not None,
                "type": type(model).__name__,
                "is_fitted": getattr(model, 'is_fitted', False)
            }

        # Cache status (Redis)
        cache_status = {"status": "unknown"}
        try:
            import redis
            from config.settings import settings
            redis_client = redis.from_url(settings.REDIS_URL)
            cache_info = redis_client.info()
            cache_status = {
                "status": "healthy",
                "used_memory": cache_info.get('used_memory_human', 'N/A'),
                "connected_clients": cache_info.get('connected_clients', 0),
                "keyspace_hits": cache_info.get('keyspace_hits', 0),
                "keyspace_misses": cache_info.get('keyspace_misses', 0)
            }
        except Exception as e:
            cache_status = {"status": "error", "error": str(e)}

        return {
            "system_health": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": db_stats,
            "recent_activity": recent_activity,
            "models": model_status,
            "cache": cache_status
        }

    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching system status"
        )


@router.post("/models/retrain")
async def retrain_models():
    """Trigger model retraining (placeholder for production implementation)"""
    try:
        # In production, this would trigger a background job to retrain models
        # with fresh data from the database

        # For now, we'll simulate the process
        logger.info("Model retraining triggered by admin")

        return {
            "message": "Model retraining initiated",
            "status": "started",
            "timestamp": datetime.now().isoformat(),
            "estimated_completion": (datetime.now() + timedelta(hours=2)).isoformat()
        }

    except Exception as e:
        logger.error(f"Error triggering model retraining: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error triggering model retraining"
        )


@router.get("/performance/summary")
async def get_performance_summary(
    days_back: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """Get system performance summary"""
    try:
        from models.database.models import DelayPrediction, CommunicationLog, Shipment
        from sqlalchemy import func, and_

        cutoff_date = datetime.now() - timedelta(days=days_back)

        # Prediction performance
        prediction_metrics = db.query(
            func.count(DelayPrediction.id).label('total_predictions'),
            func.avg(DelayPrediction.prediction_accuracy).label('avg_accuracy'),
            func.avg(DelayPrediction.confidence_score).label('avg_confidence')
        ).filter(DelayPrediction.prediction_date >= cutoff_date).first()

        # Communication performance
        comm_metrics = db.query(
            func.count(CommunicationLog.id).label('total_communications'),
            func.sum(
                func.case([(CommunicationLog.delivery_status == 'delivered', 1)], else_=0)
            ).label('successful_deliveries'),
            func.avg(CommunicationLog.engagement_score).label('avg_engagement')
        ).filter(CommunicationLog.sent_at >= cutoff_date).first()

        # Shipment performance
        shipment_metrics = db.query(
            func.count(Shipment.id).label('total_shipments'),
            func.sum(
                func.case([(Shipment.status == 'delivered', 1)], else_=0)
            ).label('delivered_shipments'),
            func.sum(
                func.case([(Shipment.status == 'delayed', 1)], else_=0)
            ).label('delayed_shipments')
        ).filter(Shipment.created_at >= cutoff_date).first()

        # Calculate rates
        prediction_accuracy = float(prediction_metrics.avg_accuracy or 0)
        prediction_confidence = float(prediction_metrics.avg_confidence or 0)

        communication_success_rate = 0
        if comm_metrics.total_communications and comm_metrics.total_communications > 0:
            communication_success_rate = (comm_metrics.successful_deliveries / comm_metrics.total_communications) * 100

        on_time_delivery_rate = 0
        if shipment_metrics.total_shipments and shipment_metrics.total_shipments > 0:
            on_time_shipments = shipment_metrics.total_shipments - (shipment_metrics.delayed_shipments or 0)
            on_time_delivery_rate = (on_time_shipments / shipment_metrics.total_shipments) * 100

        return {
            "period_days": days_back,
            "predictions": {
                "total": prediction_metrics.total_predictions or 0,
                "average_accuracy": round(prediction_accuracy, 3),
                "average_confidence": round(prediction_confidence, 3)
            },
            "communications": {
                "total": comm_metrics.total_communications or 0,
                "success_rate": round(communication_success_rate, 2),
                "successful_deliveries": comm_metrics.successful_deliveries or 0,
                "average_engagement": float(comm_metrics.avg_engagement or 0)
            },
            "shipments": {
                "total": shipment_metrics.total_shipments or 0,
                "delivered": shipment_metrics.delivered_shipments or 0,
                "delayed": shipment_metrics.delayed_shipments or 0,
                "on_time_rate": round(on_time_delivery_rate, 2)
            }
        }

    except Exception as e:
        logger.error(f"Error getting performance summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching performance summary"
        )


@router.post("/data/cleanup")
async def cleanup_old_data(
    days_to_keep: int = Query(365, ge=30, le=3650),
    dry_run: bool = Query(True),
    db: Session = Depends(get_db)
):
    """Clean up old data from database"""
    try:
        from models.database.models import ExternalData, CommunicationLog, DelayPrediction
        from sqlalchemy import and_

        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cleanup_summary = {}

        # External data cleanup (expired entries)
        expired_external_data = db.query(ExternalData).filter(
            ExternalData.expires_at < datetime.now()
        )
        expired_count = expired_external_data.count()

        if not dry_run:
            expired_external_data.delete()

        cleanup_summary['expired_external_data'] = expired_count

        # Old communication logs (keep for compliance)
        old_comm_logs = db.query(CommunicationLog).filter(
            CommunicationLog.sent_at < cutoff_date
        )
        old_comm_count = old_comm_logs.count()

        if not dry_run and days_to_keep < 2555:  # Don't delete if keeping more than 7 years
            old_comm_logs.delete()

        cleanup_summary['old_communication_logs'] = old_comm_count

        # Old predictions (keep recent ones for accuracy tracking)
        prediction_cutoff = datetime.now() - timedelta(days=min(days_to_keep, 180))  # Keep at least 6 months
        old_predictions = db.query(DelayPrediction).filter(
            and_(
                DelayPrediction.prediction_date < prediction_cutoff,
                DelayPrediction.prediction_accuracy.is_(None)  # Only remove unvalidated old predictions
            )
        )
        old_pred_count = old_predictions.count()

        if not dry_run:
            old_predictions.delete()

        cleanup_summary['old_predictions'] = old_pred_count

        if not dry_run:
            db.commit()
            logger.info(f"Data cleanup completed: {cleanup_summary}")
        else:
            logger.info(f"Data cleanup dry run: {cleanup_summary}")

        return {
            "dry_run": dry_run,
            "cleanup_summary": cleanup_summary,
            "cutoff_date": cutoff_date.isoformat(),
            "message": "Cleanup completed" if not dry_run else "Dry run completed - no data was deleted"
        }

    except Exception as e:
        logger.error(f"Error during data cleanup: {e}")
        if not dry_run:
            db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error during data cleanup"
        )


@router.get("/logs")
async def get_system_logs(
    lines: int = Query(100, ge=1, le=1000),
    level: str = Query("INFO")
):
    """Get recent system logs"""
    try:
        # In production, this would read from actual log files
        # For demo, we'll return placeholder log entries

        import random
        from datetime import datetime, timedelta

        log_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        components = ["api", "prediction", "communication", "database"]

        logs = []
        for i in range(lines):
            timestamp = datetime.now() - timedelta(minutes=random.randint(0, 1440))  # Last 24 hours
            log_level = random.choice(log_levels)
            component = random.choice(components)

            messages = {
                "INFO": [
                    f"Processed prediction for shipment {random.randint(1000, 9999)}",
                    f"Sent communication to customer {random.randint(100, 999)}",
                    f"Updated shipment status to in_transit",
                    f"Model prediction completed successfully"
                ],
                "WARNING": [
                    f"High delay probability detected for shipment {random.randint(1000, 9999)}",
                    f"External API response time elevated",
                    f"Redis cache miss rate above threshold"
                ],
                "ERROR": [
                    f"Failed to send email notification",
                    f"Database connection timeout",
                    f"Model prediction failed for invalid input"
                ]
            }

            message = random.choice(messages.get(log_level, messages["INFO"]))

            logs.append({
                "timestamp": timestamp.isoformat(),
                "level": log_level,
                "component": component,
                "message": message
            })

        # Filter by level if specified
        if level != "ALL":
            logs = [log for log in logs if log["level"] == level]

        # Sort by timestamp (newest first)
        logs.sort(key=lambda x: x["timestamp"], reverse=True)

        return {
            "logs": logs[:lines],
            "total_entries": len(logs),
            "filter_level": level
        }

    except Exception as e:
        logger.error(f"Error getting system logs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching system logs"
        )
