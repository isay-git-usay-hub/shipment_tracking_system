"""
Simple ML models for shipment delay prediction - Demo Version
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import random
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path


class SimpleDelayPredictor:
    """Simple delay prediction model for demo purposes"""
    
    def __init__(self):
        self.model_name = "SimplePredictor"
        self.is_trained = True  # Always ready for demo
    
    def predict(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Make a simple prediction based on basic rules"""
        # Simple rule-based predictions for demo
        if features_df.empty:
            return self._default_prediction()
        
        row = features_df.iloc[0]  # Take first row
        
        # Calculate base delay based on some simple factors
        base_delay = 0
        
        # Route complexity factor
        route_complexity = row.get('route_complexity', 1)
        base_delay += route_complexity * 2
        
        # Weather factor
        weather_severity = row.get('weather_severity_score', 0.5)
        base_delay += weather_severity * 12
        
        # Port congestion factor
        origin_congestion = row.get('origin_congestion', 0.5)
        destination_congestion = row.get('destination_congestion', 0.5)
        base_delay += (origin_congestion + destination_congestion) * 8
        
        # Priority factor
        is_priority = row.get('is_priority', False)
        if is_priority:
            base_delay *= 0.7  # Priority shipments have less delay
        
        # Add some randomness
        base_delay += random.gauss(0, 3)
        base_delay = max(0, base_delay)  # No negative delays
        
        # Calculate probability and confidence
        delay_probability = min(1.0, base_delay / 24.0)
        confidence_score = 0.75 + random.random() * 0.2  # 75-95%
        
        # Determine risk level
        if base_delay < 6:
            risk_level = "low"
        elif base_delay < 18:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        return {
            'predicted_delay_hours': round(base_delay, 2),
            'delay_probability': round(delay_probability, 3),
            'confidence_score': round(confidence_score, 3),
            'risk_level': risk_level,
            'model_name': self.model_name,
            'model_version': '1.0-demo',
            'factors': {
                'weather_factor': weather_severity,
                'port_congestion': (origin_congestion + destination_congestion) / 2,
                'route_complexity': route_complexity / 5.0,
                'priority_adjustment': -0.3 if is_priority else 0
            }
        }
    
    def _default_prediction(self) -> Dict[str, Any]:
        """Default prediction when no features available"""
        return {
            'predicted_delay_hours': 8.5,
            'delay_probability': 0.35,
            'confidence_score': 0.60,
            'risk_level': 'medium',
            'model_name': self.model_name,
            'model_version': '1.0-demo',
            'factors': {
                'weather_factor': 0.5,
                'port_congestion': 0.5,
                'route_complexity': 0.3,
                'priority_adjustment': 0
            }
        }


class XGBoostDelayPredictor(SimpleDelayPredictor):
    """Mock XGBoost predictor that uses simple logic"""
    
    def __init__(self):
        super().__init__()
        self.model_name = "XGBoost-Mock"
    
    def load_model(self, model_path: Path):
        """Mock model loading"""
        self.is_trained = True
        return True


class EnsembleDelayPredictor:
    """Mock ensemble predictor"""
    
    def __init__(self):
        self.models = {
            'xgboost': XGBoostDelayPredictor(),
            'simple': SimpleDelayPredictor()
        }
        self.weights = {'xgboost': 0.6, 'simple': 0.4}
    
    def load_ensemble(self, model_path: Path):
        """Mock ensemble loading"""
        return True
    
    def predict(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Ensemble prediction by averaging results"""
        predictions = []
        
        for name, model in self.models.items():
            pred = model.predict(features_df)
            predictions.append(pred)
        
        # Simple averaging for demo
        if not predictions:
            return SimpleDelayPredictor()._default_prediction()
        
        # Average the predictions
        avg_delay = sum(p['predicted_delay_hours'] for p in predictions) / len(predictions)
        avg_probability = sum(p['delay_probability'] for p in predictions) / len(predictions)
        avg_confidence = sum(p['confidence_score'] for p in predictions) / len(predictions)
        
        # Determine risk level from averaged delay
        if avg_delay < 6:
            risk_level = "low"
        elif avg_delay < 18:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        return {
            'predicted_delay_hours': round(avg_delay, 2),
            'delay_probability': round(avg_probability, 3),
            'confidence_score': round(avg_confidence, 3),
            'risk_level': risk_level,
            'model_name': 'Ensemble-Mock',
            'model_version': '1.0-demo',
            'factors': predictions[0]['factors'] if predictions else {}
        }
