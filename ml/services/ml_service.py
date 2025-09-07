"""
ML Service for Shipment Delay Prediction

This module provides ML model management, training, and prediction services.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

# Import our ML components
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from preprocessing.feature_engineering import ShipmentFeatureEngineer, load_and_prepare_data
from models.delay_predictor import (
    ModelTrainer, RandomForestDelayPredictor, 
    XGBoostDelayPredictor, LogisticRegressionDelayPredictor,
    GradientBoostingDelayPredictor, EnsembleDelayPredictor
)

logger = logging.getLogger(__name__)


class MLModelService:
    """
    Service for managing ML models, training, and predictions.
    """
    
    def __init__(self, model_dir: str = "ml/saved_models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.feature_engineer = None
        self.active_model = None
        self.active_model_name = None
        self.model_metadata = {}
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Load existing model if available
        self._load_latest_model()
        
    def _load_latest_model(self):
        """Load the most recent model if it exists."""
        try:
            model_files = list(self.model_dir.glob("*.joblib"))
            if not model_files:
                logger.info("No existing models found")
                return
                
            # Find the most recent model
            latest_model = max(model_files, key=os.path.getctime)
            model_name = latest_model.stem
            
            logger.info(f"Loading existing model: {model_name}")
            self._load_model(model_name)
            
        except Exception as e:
            logger.error(f"Error loading existing model: {e}")
    
    def _load_model(self, model_name: str) -> bool:
        """Load a specific model."""
        try:
            model_path = self.model_dir / f"{model_name}.joblib"
            metadata_path = self.model_dir / f"{model_name}_metadata.json"
            fe_path = self.model_dir / f"{model_name}_feature_engineer.joblib"
            
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # Load model based on type
            if "RandomForest" in model_name:
                self.active_model = RandomForestDelayPredictor()
            elif "XGBoost" in model_name:
                self.active_model = XGBoostDelayPredictor()
            elif "LogisticRegression" in model_name:
                self.active_model = LogisticRegressionDelayPredictor()
            elif "GradientBoosting" in model_name:
                self.active_model = GradientBoostingDelayPredictor()
            else:
                logger.error(f"Unknown model type: {model_name}")
                return False
            
            self.active_model.load_model(str(model_path))
            self.active_model_name = model_name
            
            # Load feature engineer
            if fe_path.exists():
                import joblib
                self.feature_engineer = joblib.load(fe_path)
                logger.info("Feature engineer loaded")
            else:
                logger.warning("Feature engineer not found, will need to retrain")
            
            # Load metadata
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
            
            logger.info(f"Successfully loaded model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False
    
    def _save_model(self, model, model_name: str, metadata: Dict):
        """Save a trained model with metadata."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_name}_{timestamp}"
            
            model_path = self.model_dir / f"{filename}.joblib"
            metadata_path = self.model_dir / f"{filename}_metadata.json"
            fe_path = self.model_dir / f"{filename}_feature_engineer.joblib"
            
            # Save model
            model.save_model(str(model_path))
            
            # Save feature engineer
            if self.feature_engineer:
                import joblib
                joblib.dump(self.feature_engineer, fe_path)
            
            # Save metadata
            metadata['saved_at'] = datetime.now().isoformat()
            metadata['model_file'] = str(model_path)
            metadata['feature_engineer_file'] = str(fe_path)
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Model saved: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    async def train_models(self, data_path: str, model_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Train multiple models asynchronously.
        """
        def _train_sync():
            try:
                # Load and prepare data
                logger.info(f"Loading data from {data_path}")
                df = load_and_prepare_data(data_path)
                
                # Feature engineering
                logger.info("Starting feature engineering...")
                self.feature_engineer = ShipmentFeatureEngineer()
                X_train, X_test, y_train, y_test = self.feature_engineer.create_train_test_split(df)
                
                logger.info(f"Data prepared: {len(X_train)} training samples, {len(X_test)} test samples")
                
                # Train models
                trainer = ModelTrainer()
                results = trainer.train_all_models(X_train, y_train, X_test, y_test)
                
                # Get best model
                best_model_name, best_model = trainer.get_best_model(results)
                
                # Save best model
                metadata = {
                    'training_date': datetime.now().isoformat(),
                    'dataset_size': len(df),
                    'features_count': len(self.feature_engineer.feature_columns),
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'best_model': best_model_name,
                    'all_results': {name: result.get('test_metrics', {}) for name, result in results.items()},
                    'feature_columns': self.feature_engineer.feature_columns
                }
                
                filename = self._save_model(best_model, best_model_name, metadata)
                
                # Set as active model
                self.active_model = best_model
                self.active_model_name = f"{best_model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.model_metadata = metadata
                
                return {
                    'success': True,
                    'best_model': best_model_name,
                    'filename': filename,
                    'results': results,
                    'metadata': metadata
                }
                
            except Exception as e:
                logger.error(f"Error in model training: {e}")
                return {
                    'success': False,
                    'error': str(e)
                }
        
        # Run training in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.executor, _train_sync)
        
        return result
    
    async def predict_delay(self, shipment_data: Dict) -> Dict[str, Any]:
        """
        Predict delay probability for a single shipment.
        """
        if not self.active_model or not self.feature_engineer:
            return {
                'success': False,
                'error': 'No trained model available'
            }
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame([shipment_data])
            
            # Feature engineering
            df_processed = self.feature_engineer.prepare_features(df, fit=False)
            X, _ = self.feature_engineer.get_features_and_target(df_processed)
            
            # Make prediction
            prediction = self.active_model.predict(X)[0]
            probability = self.active_model.predict_proba(X)[0]
            
            # Risk assessment
            delay_probability = probability[1]  # Probability of delay
            
            if delay_probability > 0.8:
                risk_level = "High"
            elif delay_probability > 0.5:
                risk_level = "Medium"
            elif delay_probability > 0.2:
                risk_level = "Low"
            else:
                risk_level = "Very Low"
            
            # Feature importance for this prediction
            feature_importance = self.active_model.get_feature_importance()
            if feature_importance:
                top_factors = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            else:
                top_factors = []
            
            return {
                'success': True,
                'prediction': {
                    'will_delay': bool(prediction),
                    'delay_probability': float(delay_probability),
                    'no_delay_probability': float(probability[0]),
                    'risk_level': risk_level,
                    'confidence': float(max(probability)),
                    'top_risk_factors': top_factors,
                    'model_used': self.active_model_name
                }
            }
            
        except Exception as e:
            logger.error(f"Error in delay prediction: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def predict_batch(self, shipments: List[Dict]) -> Dict[str, Any]:
        """
        Predict delays for multiple shipments.
        """
        if not self.active_model or not self.feature_engineer:
            return {
                'success': False,
                'error': 'No trained model available'
            }
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(shipments)
            
            # Feature engineering
            df_processed = self.feature_engineer.prepare_features(df, fit=False)
            X, _ = self.feature_engineer.get_features_and_target(df_processed)
            
            # Make predictions
            predictions = self.active_model.predict(X)
            probabilities = self.active_model.predict_proba(X)
            
            results = []
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                delay_probability = prob[1]
                
                if delay_probability > 0.8:
                    risk_level = "High"
                elif delay_probability > 0.5:
                    risk_level = "Medium" 
                elif delay_probability > 0.2:
                    risk_level = "Low"
                else:
                    risk_level = "Very Low"
                
                results.append({
                    'shipment_index': i,
                    'will_delay': bool(pred),
                    'delay_probability': float(delay_probability),
                    'risk_level': risk_level,
                    'confidence': float(max(prob))
                })
            
            # Summary statistics
            total_shipments = len(results)
            predicted_delays = sum(1 for r in results if r['will_delay'])
            avg_delay_prob = np.mean([r['delay_probability'] for r in results])
            high_risk_count = sum(1 for r in results if r['risk_level'] == 'High')
            
            return {
                'success': True,
                'predictions': results,
                'summary': {
                    'total_shipments': total_shipments,
                    'predicted_delays': predicted_delays,
                    'delay_rate': predicted_delays / total_shipments if total_shipments > 0 else 0,
                    'average_delay_probability': float(avg_delay_prob),
                    'high_risk_shipments': high_risk_count,
                    'model_used': self.active_model_name
                }
            }
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the active model.
        """
        if not self.active_model:
            return {
                'active': False,
                'message': 'No active model'
            }
        
        return {
            'active': True,
            'model_name': self.active_model_name,
            'model_type': self.active_model.model_name,
            'is_trained': self.active_model.is_trained,
            'metadata': self.model_metadata,
            'feature_count': len(self.feature_engineer.feature_columns) if self.feature_engineer else 0,
            'training_metrics': self.active_model.training_metrics
        }
    
    def list_saved_models(self) -> List[Dict[str, str]]:
        """
        List all saved models.
        """
        models = []
        
        try:
            for model_file in self.model_dir.glob("*.joblib"):
                if "_feature_engineer" in model_file.name:
                    continue
                    
                metadata_file = self.model_dir / f"{model_file.stem}_metadata.json"
                
                model_info = {
                    'filename': model_file.stem,
                    'filepath': str(model_file),
                    'created_at': datetime.fromtimestamp(model_file.stat().st_ctime).isoformat(),
                    'size': model_file.stat().st_size
                }
                
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            model_info['metadata'] = metadata
                    except Exception as e:
                        logger.warning(f"Error reading metadata for {model_file.stem}: {e}")
                
                models.append(model_info)
                
        except Exception as e:
            logger.error(f"Error listing models: {e}")
        
        return sorted(models, key=lambda x: x['created_at'], reverse=True)
    
    async def evaluate_model(self, data_path: str) -> Dict[str, Any]:
        """
        Evaluate the active model on new data.
        """
        if not self.active_model or not self.feature_engineer:
            return {
                'success': False,
                'error': 'No trained model available'
            }
        
        def _evaluate_sync():
            try:
                # Load evaluation data
                df = load_and_prepare_data(data_path)
                
                # Feature engineering
                df_processed = self.feature_engineer.prepare_features(df, fit=False)
                X, y = self.feature_engineer.get_features_and_target(df_processed)
                
                if y is None:
                    return {
                        'success': False,
                        'error': 'No target variable found in evaluation data'
                    }
                
                # Evaluate model
                metrics = self.active_model.evaluate(X, y)
                
                return {
                    'success': True,
                    'metrics': metrics,
                    'evaluation_size': len(X),
                    'model_name': self.active_model_name
                }
                
            except Exception as e:
                logger.error(f"Error in model evaluation: {e}")
                return {
                    'success': False,
                    'error': str(e)
                }
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.executor, _evaluate_sync)
        
        return result
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """
        Get feature importance from the active model.
        """
        if not self.active_model:
            return {
                'success': False,
                'error': 'No active model'
            }
        
        importance = self.active_model.get_feature_importance()
        if not importance:
            return {
                'success': False,
                'error': 'Feature importance not available for this model'
            }
        
        # Sort by importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'success': True,
            'feature_importance': sorted_features,
            'model_name': self.active_model_name
        }
    
    async def retrain_model(self, data_path: str) -> Dict[str, Any]:
        """
        Retrain the active model with new data.
        """
        return await self.train_models(data_path)
    
    def cleanup(self):
        """
        Cleanup resources.
        """
        if hasattr(self.executor, 'shutdown'):
            self.executor.shutdown(wait=True)


# Global ML service instance
_ml_service_instance = None

def get_ml_service() -> MLModelService:
    """
    Get or create the ML service singleton instance.
    """
    global _ml_service_instance
    if _ml_service_instance is None:
        _ml_service_instance = MLModelService()
    return _ml_service_instance


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def test_ml_service():
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        
        # Create ML service
        ml_service = MLModelService()
        
        # Train models
        print("Training models...")
        result = await ml_service.train_models("../../data/smart_logistics_dataset.csv")
        
        if result['success']:
            print(f"Training successful. Best model: {result['best_model']}")
            
            # Test single prediction
            test_shipment = {
                'asset_id': 'Truck_1',
                'latitude': 40.7128,
                'longitude': -74.0060,
                'distance_traveled': 250.5,
                'fuel_efficiency': 12.5,
                'waiting_time': 30,
                'status': 'In Transit',
                'traffic_conditions': 'Heavy',
                'weather_conditions': 'Rainy',
                'delay_reason': 'Weather',
                'timestamp': '2024-01-15 14:30:00'
            }
            
            prediction = await ml_service.predict_delay(test_shipment)
            if prediction['success']:
                pred_result = prediction['prediction']
                print(f"\\nDelay prediction:")
                print(f"  Will delay: {pred_result['will_delay']}")
                print(f"  Probability: {pred_result['delay_probability']:.3f}")
                print(f"  Risk level: {pred_result['risk_level']}")
                
            # Get model info
            info = ml_service.get_model_info()
            print(f"\\nModel info: {info}")
            
        else:
            print(f"Training failed: {result['error']}")
        
        # Cleanup
        ml_service.cleanup()
    
    # Run test
    asyncio.run(test_ml_service())
