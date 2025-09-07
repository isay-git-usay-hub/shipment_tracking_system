"""
Prediction endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List

from models.database.connection import get_db
from models.schemas import (
    PredictionRequest, PredictionResponse,
    DelayPredictionResponse
)
from services.prediction.service import prediction_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/delay", response_model=PredictionResponse)
async def predict_delay(request: PredictionRequest):
    """Predict shipment delay"""
    try:
        prediction = await prediction_service.predict_delay(request)
        return prediction

    except Exception as e:
        logger.error(f"Error in delay prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating delay prediction"
        )


@router.post("/batch", response_model=List[PredictionResponse])
async def batch_predict_delays(requests: List[PredictionRequest]):
    """Batch predict delays for multiple shipments"""
    try:
        predictions = await prediction_service.batch_predict(requests)
        return predictions

    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating batch predictions"
        )


@router.get("/shipment/{shipment_id}", response_model=List[DelayPredictionResponse])
async def get_shipment_predictions(
    shipment_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db)
):
    """Get prediction history for a shipment"""
    try:
        from models.database.models import DelayPrediction, Shipment

        # Verify shipment exists
        shipment = db.query(Shipment).filter(Shipment.id == shipment_id).first()
        if not shipment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Shipment not found"
            )

        predictions = db.query(DelayPrediction).filter(
            DelayPrediction.shipment_id == shipment_id
        ).order_by(DelayPrediction.prediction_date.desc()).offset(skip).limit(limit).all()

        return predictions

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching predictions for shipment {shipment_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching predictions"
        )


@router.get("/accuracy")
async def get_prediction_accuracy(days_back: int = Query(30, ge=1, le=365)):
    """Get prediction accuracy metrics"""
    try:
        accuracy_metrics = await prediction_service.get_prediction_accuracy(days_back)
        return accuracy_metrics

    except Exception as e:
        logger.error(f"Error calculating prediction accuracy: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error calculating accuracy metrics"
        )


@router.get("/models/status")
async def get_model_status():
    """Get status of prediction models"""
    try:
        models_info = {}
        for name, model in prediction_service.models.items():
            models_info[name] = {
                "loaded": model is not None,
                "type": type(model).__name__,
                "is_fitted": getattr(model, 'is_fitted', False)
            }

        return {
            "models": models_info,
            "active_models": len([m for m in models_info.values() if m["loaded"]]),
            "status": "healthy" if models_info else "no_models_loaded"
        }

    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error getting model status"
        )


@router.post("/models/reload")
async def reload_models():
    """Reload prediction models"""
    try:
        prediction_service._load_models()
        return {"message": "Models reloaded successfully"}

    except Exception as e:
        logger.error(f"Error reloading models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error reloading models"
        )
