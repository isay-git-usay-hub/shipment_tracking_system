"""
Dashboard endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime, timedelta

from models.database.connection import get_db
from models.schemas import DashboardStats, DashboardShipment
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/stats", response_model=DashboardStats)
async def get_dashboard_stats(db: Session = Depends(get_db)):
    """Get dashboard statistics"""
    try:
        from models.database.models import Shipment, Customer, DelayPrediction, CommunicationLog
        from sqlalchemy import func, and_

        # Total shipments
        total_shipments = db.query(Shipment).count()

        # Active shipments (in transit, pending)
        active_shipments = db.query(Shipment).filter(
            Shipment.status.in_(['pending', 'in_transit'])
        ).count()

        # Delayed shipments
        delayed_shipments = db.query(Shipment).filter(
            Shipment.status == 'delayed'
        ).count()

        # On-time shipments (delivered on schedule)
        on_time_shipments = db.query(Shipment).filter(
            and_(
                Shipment.status == 'delivered',
                Shipment.actual_arrival <= Shipment.scheduled_arrival
            )
        ).count()

        # Total customers
        total_customers = db.query(Customer).filter(Customer.is_active == True).count()

        # Predictions today
        today = datetime.now().date()
        predictions_today = db.query(DelayPrediction).filter(
            func.date(DelayPrediction.prediction_date) == today
        ).count()

        # Communications sent today
        communications_today = db.query(CommunicationLog).filter(
            func.date(CommunicationLog.sent_at) == today
        ).count()

        # Average delay hours (for completed shipments with delays)
        avg_delay_query = db.query(
            func.avg(
                func.extract('epoch', Shipment.actual_arrival - Shipment.scheduled_arrival) / 3600
            ).label('avg_delay')
        ).filter(
            and_(
                Shipment.actual_arrival.isnot(None),
                Shipment.actual_arrival > Shipment.scheduled_arrival
            )
        ).scalar()

        avg_delay_hours = float(avg_delay_query) if avg_delay_query else 0.0

        # Prediction accuracy (for predictions with actual results)
        accuracy_query = db.query(
            func.avg(DelayPrediction.prediction_accuracy).label('avg_accuracy')
        ).filter(
            DelayPrediction.prediction_accuracy.isnot(None)
        ).scalar()

        prediction_accuracy = float(accuracy_query) if accuracy_query else 0.0

        return DashboardStats(
            total_shipments=total_shipments,
            active_shipments=active_shipments,
            delayed_shipments=delayed_shipments,
            on_time_shipments=on_time_shipments,
            total_customers=total_customers,
            predictions_today=predictions_today,
            communications_sent_today=communications_today,
            average_delay_hours=avg_delay_hours,
            delay_prediction_accuracy=prediction_accuracy
        )

    except Exception as e:
        logger.error(f"Error getting dashboard stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching dashboard statistics"
        )


@router.get("/shipments/recent", response_model=List[DashboardShipment])
async def get_recent_shipments(
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Get recent shipments for dashboard"""
    try:
        from models.database.models import Shipment, Customer, DelayPrediction
        from sqlalchemy.orm import joinedload

        # Get recent shipments with customer and latest prediction
        shipments = db.query(Shipment).options(
            joinedload(Shipment.customer)
        ).order_by(Shipment.created_at.desc()).limit(limit).all()

        dashboard_shipments = []
        for shipment in shipments:
            # Get latest prediction
            latest_prediction = db.query(DelayPrediction).filter(
                DelayPrediction.shipment_id == shipment.id
            ).order_by(DelayPrediction.prediction_date.desc()).first()

            # Calculate risk level
            risk_level = None
            if latest_prediction:
                if latest_prediction.predicted_delay_hours >= 24:
                    risk_level = "high"
                elif latest_prediction.predicted_delay_hours >= 8:
                    risk_level = "medium"
                else:
                    risk_level = "low"

            dashboard_shipment = DashboardShipment(
                id=shipment.id,
                shipment_id=shipment.shipment_id,
                customer_name=shipment.customer.name if shipment.customer else "Unknown",
                origin_port=shipment.origin_port,
                destination_port=shipment.destination_port,
                status=shipment.status,
                scheduled_arrival=shipment.scheduled_arrival,
                estimated_arrival=shipment.estimated_arrival,
                delay_probability=latest_prediction.delay_probability if latest_prediction else None,
                predicted_delay_hours=latest_prediction.predicted_delay_hours if latest_prediction else None,
                risk_level=risk_level
            )
            dashboard_shipments.append(dashboard_shipment)

        return dashboard_shipments

    except Exception as e:
        logger.error(f"Error getting recent shipments: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching recent shipments"
        )


@router.get("/shipments/high-risk", response_model=List[DashboardShipment])
async def get_high_risk_shipments(
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Get high-risk shipments for dashboard"""
    try:
        from models.database.models import Shipment, Customer, DelayPrediction
        from sqlalchemy.orm import joinedload

        # Get shipments with high delay probability
        high_risk_predictions = db.query(DelayPrediction).filter(
            DelayPrediction.delay_probability >= 0.7
        ).order_by(DelayPrediction.delay_probability.desc()).limit(limit).all()

        dashboard_shipments = []
        for prediction in high_risk_predictions:
            shipment = db.query(Shipment).options(
                joinedload(Shipment.customer)
            ).filter(Shipment.id == prediction.shipment_id).first()

            if shipment and shipment.status in ['pending', 'in_transit']:
                dashboard_shipment = DashboardShipment(
                    id=shipment.id,
                    shipment_id=shipment.shipment_id,
                    customer_name=shipment.customer.name if shipment.customer else "Unknown",
                    origin_port=shipment.origin_port,
                    destination_port=shipment.destination_port,
                    status=shipment.status,
                    scheduled_arrival=shipment.scheduled_arrival,
                    estimated_arrival=shipment.estimated_arrival,
                    delay_probability=prediction.delay_probability,
                    predicted_delay_hours=prediction.predicted_delay_hours,
                    risk_level="high"
                )
                dashboard_shipments.append(dashboard_shipment)

        return dashboard_shipments

    except Exception as e:
        logger.error(f"Error getting high-risk shipments: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching high-risk shipments"
        )


@router.get("/analytics/delays")
async def get_delay_analytics(
    days_back: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """Get delay analytics for dashboard charts"""
    try:
        from models.database.models import Shipment, DelayPrediction
        from sqlalchemy import func, and_

        cutoff_date = datetime.now() - timedelta(days=days_back)

        # Daily delay trends
        daily_delays = db.query(
            func.date(Shipment.created_at).label('date'),
            func.count(Shipment.id).label('total_shipments'),
            func.sum(
                func.case([(Shipment.status == 'delayed', 1)], else_=0)
            ).label('delayed_shipments')
        ).filter(
            Shipment.created_at >= cutoff_date
        ).group_by(
            func.date(Shipment.created_at)
        ).order_by('date').all()

        # Delay by route
        route_delays = db.query(
            Shipment.origin_port,
            Shipment.destination_port,
            func.count(Shipment.id).label('total_shipments'),
            func.avg(
                func.extract('epoch', Shipment.actual_arrival - Shipment.scheduled_arrival) / 3600
            ).label('avg_delay_hours')
        ).filter(
            and_(
                Shipment.created_at >= cutoff_date,
                Shipment.actual_arrival.isnot(None)
            )
        ).group_by(
            Shipment.origin_port,
            Shipment.destination_port
        ).having(
            func.count(Shipment.id) >= 5  # Only routes with at least 5 shipments
        ).order_by('avg_delay_hours desc').limit(10).all()

        # Prediction accuracy over time
        prediction_accuracy = db.query(
            func.date(DelayPrediction.prediction_date).label('date'),
            func.avg(DelayPrediction.prediction_accuracy).label('avg_accuracy')
        ).filter(
            and_(
                DelayPrediction.prediction_date >= cutoff_date,
                DelayPrediction.prediction_accuracy.isnot(None)
            )
        ).group_by(
            func.date(DelayPrediction.prediction_date)
        ).order_by('date').all()

        return {
            "daily_trends": [
                {
                    "date": str(row.date),
                    "total_shipments": row.total_shipments,
                    "delayed_shipments": row.delayed_shipments,
                    "delay_rate": (row.delayed_shipments / row.total_shipments * 100) if row.total_shipments > 0 else 0
                }
                for row in daily_delays
            ],
            "route_performance": [
                {
                    "route": f"{row.origin_port} → {row.destination_port}",
                    "total_shipments": row.total_shipments,
                    "avg_delay_hours": float(row.avg_delay_hours) if row.avg_delay_hours else 0
                }
                for row in route_delays
            ],
            "prediction_accuracy": [
                {
                    "date": str(row.date),
                    "accuracy": float(row.avg_accuracy) if row.avg_accuracy else 0
                }
                for row in prediction_accuracy
            ]
        }

    except Exception as e:
        logger.error(f"Error getting delay analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching delay analytics"
        )


@router.get("/alerts")
async def get_dashboard_alerts(db: Session = Depends(get_db)):
    """Get alerts for dashboard"""
    try:
        from models.database.models import Shipment, DelayPrediction, CommunicationLog
        from sqlalchemy import and_, func

        alerts = []

        # High-risk shipments alert
        high_risk_count = db.query(DelayPrediction).join(Shipment).filter(
            and_(
                DelayPrediction.delay_probability >= 0.8,
                Shipment.status.in_(['pending', 'in_transit'])
            )
        ).count()

        if high_risk_count > 0:
            alerts.append({
                "type": "high_risk",
                "severity": "high",
                "message": f"{high_risk_count} shipments have high delay risk (>80%)",
                "count": high_risk_count
            })

        # Overdue shipments alert
        overdue_count = db.query(Shipment).filter(
            and_(
                Shipment.scheduled_arrival < datetime.now(),
                Shipment.status.in_(['pending', 'in_transit'])
            )
        ).count()

        if overdue_count > 0:
            alerts.append({
                "type": "overdue",
                "severity": "critical",
                "message": f"{overdue_count} shipments are overdue",
                "count": overdue_count
            })

        # Failed communications alert
        today = datetime.now().date()
        failed_comms = db.query(CommunicationLog).filter(
            and_(
                func.date(CommunicationLog.sent_at) == today,
                CommunicationLog.delivery_status == 'failed'
            )
        ).count()

        if failed_comms > 0:
            alerts.append({
                "type": "failed_communications",
                "severity": "medium",
                "message": f"{failed_comms} communications failed to deliver today",
                "count": failed_comms
            })

        return {"alerts": alerts, "total_alerts": len(alerts)}

    except Exception as e:
        logger.error(f"Error getting dashboard alerts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching alerts"
        )
