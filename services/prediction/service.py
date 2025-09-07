"""
Prediction service for shipment delay prediction
"""
import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from models.database.connection import get_db
from models.database.models import Shipment, DelayPrediction, Customer, ExternalData
from models.ml_models.simple_predictor import EnsembleDelayPredictor, XGBoostDelayPredictor
from models.schemas import PredictionRequest, PredictionResponse
from services.external_data import WeatherService, PortService
from config.settings import settings
import redis

logger = logging.getLogger(__name__)


class PredictionService:
    """Service for handling delay predictions"""

    def __init__(self):
        self.models = {}
        self.redis_client = redis.from_url(settings.REDIS_URL)
        self.weather_service = WeatherService()
        self.port_service = PortService()
        self.model_path = Path(settings.MODEL_PATH)
        self._load_models()

    def _load_models(self) -> None:
        """Load trained models"""
        try:
            # Load ensemble model
            ensemble_path = self.model_path / "ensemble"
            if ensemble_path.exists():
                self.models['ensemble'] = EnsembleDelayPredictor()
                self.models['ensemble'].load_ensemble(ensemble_path)
                logger.info("Loaded ensemble model")

            # Load XGBoost as fallback
            xgb_path = self.model_path / "xgboost_model.pkl"
            if xgb_path.exists():
                self.models['xgboost'] = XGBoostDelayPredictor()
                self.models['xgboost'].load_model(xgb_path)
                logger.info("Loaded XGBoost model")

            if not self.models:
                logger.warning("No trained models found. Using default model.")
                self.models['default'] = XGBoostDelayPredictor()

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.models['default'] = XGBoostDelayPredictor()

    async def predict_delay(self, request: PredictionRequest) -> PredictionResponse:
        """Predict delay for a shipment"""
        try:
            # Prepare features
            features_df = await self._prepare_features(request)

            # Get cached prediction if available
            cache_key = self._generate_cache_key(request)
            cached_result = self._get_cached_prediction(cache_key)
            if cached_result:
                logger.info(f"Returning cached prediction for {cache_key}")
                return cached_result

            # Make prediction
            prediction_result = await self._make_prediction(features_df, request)

            # Cache result
            self._cache_prediction(cache_key, prediction_result)

            # Store prediction in database if shipment_id provided
            if request.shipment_id:
                await self._store_prediction(request.shipment_id, prediction_result, features_df)

            return prediction_result

        except Exception as e:
            logger.error(f"Error in delay prediction: {e}")
            # Return safe default prediction
            return PredictionResponse(
                shipment_id=request.shipment_id,
                predicted_delay_hours=0.0,
                delay_probability=0.1,
                confidence_score=0.1,
                risk_level="unknown",
                factors={},
                recommendations=["Unable to generate prediction. Please check manually."],
                model_info={"error": str(e)},
                prediction_timestamp=datetime.now()
            )

    async def _prepare_features(self, request: PredictionRequest) -> pd.DataFrame:
        """Prepare features for prediction"""
        # Base features from request
        features = {
            'origin_port': request.origin_port,
            'destination_port': request.destination_port,
            'scheduled_departure': request.scheduled_departure,
            'scheduled_arrival': request.scheduled_arrival,
            'cargo_weight': request.cargo_weight or 0,
            'cargo_type': request.cargo_type or 'general',
            'container_type': request.container_type or 'standard',
            'route': request.route or f"{request.origin_port}-{request.destination_port}",
            'is_priority': request.is_priority or False
        }

        # Add external data
        external_data = await self._fetch_external_data(request)
        features.update(external_data)

        # Add historical data
        historical_data = await self._fetch_historical_data(request)
        features.update(historical_data)

        return pd.DataFrame([features])

    async def _fetch_external_data(self, request: PredictionRequest) -> Dict[str, Any]:
        """Fetch external data (weather, port congestion, etc.)"""
        external_data = {}

        try:
            # Weather data
            if request.weather_conditions:
                weather_data = request.weather_conditions
            else:
                weather_data = await self.weather_service.get_weather_data(
                    request.origin_port, request.scheduled_departure
                )

            external_data.update({
                'weather_severity_score': self._calculate_weather_severity(weather_data),
                'temperature': weather_data.get('temperature', 20),
                'wind_speed': weather_data.get('wind_speed', 5),
                'precipitation': weather_data.get('precipitation', 0)
            })

            # Port congestion data
            if request.port_congestion_data:
                port_data = request.port_congestion_data
            else:
                origin_congestion = await self.port_service.get_port_congestion(request.origin_port)
                destination_congestion = await self.port_service.get_port_congestion(request.destination_port)
                port_data = {
                    'origin_congestion': origin_congestion.get('congestion_level', 0.5),
                    'destination_congestion': destination_congestion.get('congestion_level', 0.5)
                }

            external_data.update(port_data)

        except Exception as e:
            logger.warning(f"Error fetching external data: {e}")
            # Use default values
            external_data.update({
                'weather_severity_score': 0.5,
                'temperature': 20,
                'wind_speed': 5,
                'precipitation': 0,
                'origin_congestion': 0.5,
                'destination_congestion': 0.5
            })

        return external_data

    async def _fetch_historical_data(self, request: PredictionRequest) -> Dict[str, Any]:
        """Fetch historical performance data"""
        historical_data = {}

        try:
            db = next(get_db())

            # Get historical performance for this route
            route_shipments = db.query(Shipment).filter(
                and_(
                    Shipment.origin_port == request.origin_port,
                    Shipment.destination_port == request.destination_port,
                    Shipment.actual_arrival.isnot(None)
                )
            ).limit(100).all()

            if route_shipments:
                delays = []
                for shipment in route_shipments:
                    if shipment.actual_arrival and shipment.scheduled_arrival:
                        delay_hours = (shipment.actual_arrival - shipment.scheduled_arrival).total_seconds() / 3600
                        delays.append(delay_hours)

                if delays:
                    historical_data.update({
                        'route_avg_delay': np.mean(delays),
                        'route_median_delay': np.median(delays),
                        'route_delay_std': np.std(delays),
                        'route_max_delay': np.max(delays),
                        'route_frequency': len(delays)
                    })

            # Default values if no historical data
            if not historical_data:
                historical_data.update({
                    'route_avg_delay': 0,
                    'route_median_delay': 0,
                    'route_delay_std': 0,
                    'route_max_delay': 0,
                    'route_frequency': 0
                })

            db.close()

        except Exception as e:
            logger.warning(f"Error fetching historical data: {e}")
            historical_data.update({
                'route_avg_delay': 0,
                'route_median_delay': 0,
                'route_delay_std': 0,
                'route_max_delay': 0,
                'route_frequency': 0
            })

        return historical_data

    async def _make_prediction(self, features_df: pd.DataFrame, request: PredictionRequest) -> PredictionResponse:
        """Make the actual prediction"""
        # Choose best available model
        model_name = 'ensemble' if 'ensemble' in self.models else list(self.models.keys())[0]
        model = self.models[model_name]

        try:
            if model_name == 'ensemble':
                result = model.predict(features_df)
                predicted_delay = result['ensemble'][0]
                confidence = result['confidence'][0]
                individual_predictions = result['individual']
            else:
                predicted_delay = model.predict(features_df)[0]
                confidence = 0.7  # Default confidence
                individual_predictions = {model_name: [predicted_delay]}

        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            predicted_delay = 0.0
            confidence = 0.1
            individual_predictions = {}

        # Calculate delay probability and risk level
        delay_probability = min(predicted_delay / 24.0, 1.0)  # Normalize by 24 hours
        risk_level = self._calculate_risk_level(predicted_delay, delay_probability)

        # Generate factors explanation
        factors = self._generate_factors_explanation(features_df.iloc[0], predicted_delay)

        # Generate recommendations
        recommendations = self._generate_recommendations(predicted_delay, factors, request)

        return PredictionResponse(
            shipment_id=request.shipment_id,
            predicted_delay_hours=float(predicted_delay),
            delay_probability=float(delay_probability),
            confidence_score=float(confidence),
            risk_level=risk_level,
            factors=factors,
            recommendations=recommendations,
            model_info={
                "model_name": model_name,
                "version": "1.0",
                "individual_predictions": {k: float(v[0]) if v else 0.0 for k, v in individual_predictions.items()}
            },
            prediction_timestamp=datetime.now()
        )

    def _calculate_weather_severity(self, weather_data: Dict[str, Any]) -> float:
        """Calculate weather severity score"""
        severity = 0.0

        # Temperature extremes
        temp = weather_data.get('temperature', 20)
        if temp < 0 or temp > 40:
            severity += 0.3

        # Wind speed
        wind = weather_data.get('wind_speed', 0)
        if wind > 25:  # Strong winds
            severity += 0.4

        # Precipitation
        precip = weather_data.get('precipitation', 0)
        if precip > 5:  # Heavy rain/snow
            severity += 0.3

        return min(severity, 1.0)

    def _calculate_risk_level(self, predicted_delay: float, delay_probability: float) -> str:
        """Calculate risk level based on predicted delay and probability"""
        if predicted_delay >= 24 and delay_probability >= 0.7:
            return "high"
        elif predicted_delay >= 8 and delay_probability >= 0.5:
            return "medium"
        elif predicted_delay >= 2 and delay_probability >= 0.3:
            return "low"
        else:
            return "minimal"

    def _generate_factors_explanation(self, features: pd.Series, predicted_delay: float) -> Dict[str, float]:
        """Generate explanation of factors contributing to delay"""
        factors = {}

        # Weather factor
        weather_score = features.get('weather_severity_score', 0)
        factors['weather'] = float(weather_score * 0.3)

        # Port congestion factor
        origin_congestion = features.get('origin_congestion', 0)
        dest_congestion = features.get('destination_congestion', 0)
        factors['port_congestion'] = float((origin_congestion + dest_congestion) / 2 * 0.4)

        # Route history factor
        historical_delay = features.get('route_avg_delay', 0)
        factors['route_history'] = float(min(historical_delay / 24, 0.5))

        # Cargo factor
        cargo_weight = features.get('cargo_weight', 0)
        if cargo_weight > 20000:  # Heavy cargo
            factors['cargo_complexity'] = 0.2
        else:
            factors['cargo_complexity'] = 0.0

        # Priority factor
        if features.get('is_priority', False):
            factors['priority_handling'] = -0.1  # Priority reduces delay risk
        else:
            factors['priority_handling'] = 0.0

        return factors

    def _generate_recommendations(self, predicted_delay: float, factors: Dict[str, float], 
                                request: PredictionRequest) -> List[str]:
        """Generate recommendations based on prediction"""
        recommendations = []

        if predicted_delay >= 24:
            recommendations.append("High delay risk detected. Consider alternative routes or expedited shipping.")
            recommendations.append("Proactively communicate with customer about potential delays.")
        elif predicted_delay >= 8:
            recommendations.append("Moderate delay risk. Monitor shipment closely.")
            recommendations.append("Consider notifying customer of potential delay.")

        # Weather-specific recommendations
        if factors.get('weather', 0) > 0.3:
            recommendations.append("Severe weather conditions detected. Consider delaying departure if possible.")

        # Port congestion recommendations
        if factors.get('port_congestion', 0) > 0.5:
            recommendations.append("High port congestion detected. Consider alternative ports or routes.")

        # Route history recommendations
        if factors.get('route_history', 0) > 0.3:
            recommendations.append("This route has historical delay patterns. Add buffer time to schedule.")

        if not recommendations:
            recommendations.append("Low delay risk. Proceed with standard handling procedures.")

        return recommendations

    def _generate_cache_key(self, request: PredictionRequest) -> str:
        """Generate cache key for prediction"""
        key_data = {
            'origin': request.origin_port,
            'destination': request.destination_port,
            'departure_date': request.scheduled_departure.date().isoformat(),
            'cargo_weight': request.cargo_weight or 0,
            'is_priority': request.is_priority or False
        }
        return f"prediction:{hash(json.dumps(key_data, sort_keys=True))}"

    def _get_cached_prediction(self, cache_key: str) -> Optional[PredictionResponse]:
        """Get cached prediction"""
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                data = json.loads(cached_data)
                return PredictionResponse(**data)
        except Exception as e:
            logger.warning(f"Error retrieving cached prediction: {e}")
        return None

    def _cache_prediction(self, cache_key: str, prediction: PredictionResponse) -> None:
        """Cache prediction result"""
        try:
            # Convert to dict for JSON serialization
            data = prediction.model_dump()
            data['prediction_timestamp'] = prediction.prediction_timestamp.isoformat()

            self.redis_client.setex(
                cache_key,
                settings.MODEL_CACHE_TTL_MINUTES * 60,
                json.dumps(data)
            )
        except Exception as e:
            logger.warning(f"Error caching prediction: {e}")

    async def _store_prediction(self, shipment_id: int, prediction: PredictionResponse, 
                              features_df: pd.DataFrame) -> None:
        """Store prediction in database"""
        try:
            db = next(get_db())

            # Extract model info
            model_info = prediction.model_info
            model_name = model_info.get('model_name', 'unknown')
            model_version = model_info.get('version', '1.0')

            # Create prediction record
            db_prediction = DelayPrediction(
                shipment_id=shipment_id,
                predicted_delay_hours=prediction.predicted_delay_hours,
                delay_probability=prediction.delay_probability,
                confidence_score=prediction.confidence_score,
                model_name=model_name,
                model_version=model_version,
                features_used=features_df.iloc[0].to_dict(),
                weather_factor=prediction.factors.get('weather', 0),
                port_congestion_factor=prediction.factors.get('port_congestion', 0),
                route_complexity_factor=prediction.factors.get('route_history', 0),
                prediction_date=prediction.prediction_timestamp
            )

            db.add(db_prediction)
            db.commit()
            db.close()

            logger.info(f"Stored prediction for shipment {shipment_id}")

        except Exception as e:
            logger.error(f"Error storing prediction: {e}")

    async def batch_predict(self, requests: List[PredictionRequest]) -> List[PredictionResponse]:
        """Batch prediction for multiple shipments"""
        tasks = [self.predict_delay(request) for request in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and log them
        predictions = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch prediction failed for request {i}: {result}")
                # Add default prediction
                predictions.append(PredictionResponse(
                    shipment_id=requests[i].shipment_id,
                    predicted_delay_hours=0.0,
                    delay_probability=0.1,
                    confidence_score=0.1,
                    risk_level="unknown",
                    factors={},
                    recommendations=["Prediction failed"],
                    model_info={"error": str(result)},
                    prediction_timestamp=datetime.now()
                ))
            else:
                predictions.append(result)

        return predictions

    async def get_prediction_accuracy(self, days_back: int = 30) -> Dict[str, float]:
        """Calculate prediction accuracy for recent predictions"""
        try:
            db = next(get_db())

            # Get predictions with actual results
            cutoff_date = datetime.now() - timedelta(days=days_back)
            predictions = db.query(DelayPrediction).join(Shipment).filter(
                and_(
                    DelayPrediction.prediction_date >= cutoff_date,
                    Shipment.actual_arrival.isnot(None),
                    DelayPrediction.actual_delay_hours.isnot(None)
                )
            ).all()

            if not predictions:
                return {"message": "No predictions with actual results found"}

            # Calculate metrics
            actual_delays = [p.actual_delay_hours for p in predictions]
            predicted_delays = [p.predicted_delay_hours for p in predictions]

            mae = np.mean(np.abs(np.array(actual_delays) - np.array(predicted_delays)))
            rmse = np.sqrt(np.mean((np.array(actual_delays) - np.array(predicted_delays)) ** 2))

            # Calculate accuracy for binary classification (delayed vs on-time)
            actual_delayed = [1 if delay > 2 else 0 for delay in actual_delays]
            predicted_delayed = [1 if delay > 2 else 0 for delay in predicted_delays]
            accuracy = np.mean(np.array(actual_delayed) == np.array(predicted_delayed))

            db.close()

            return {
                "predictions_evaluated": len(predictions),
                "mae_hours": float(mae),
                "rmse_hours": float(rmse),
                "classification_accuracy": float(accuracy),
                "days_evaluated": days_back
            }

        except Exception as e:
            logger.error(f"Error calculating prediction accuracy: {e}")
            return {"error": str(e)}


# Global prediction service instance
prediction_service = PredictionService()
