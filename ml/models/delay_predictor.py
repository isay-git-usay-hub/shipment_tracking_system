"""
Delay Prediction Models for Shipment AI System

This module implements various ML models for predicting shipment delays.
"""

import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from datetime import datetime

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.model_selection import cross_val_score, GridSearchCV

# Try to import XGBoost (optional)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Install with: pip install xgboost")

logger = logging.getLogger(__name__)


class DelayPredictionModel:
    """
    Base class for delay prediction models.
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.feature_importance = None
        self.training_metrics = {}
        self.model_path = None
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, float]:
        """Train the model and return metrics."""
        raise NotImplementedError
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For models without predict_proba, return decision function or binary predictions
            predictions = self.predict(X)
            proba = np.zeros((len(predictions), 2))
            proba[:, 1] = predictions  # Positive class
            proba[:, 0] = 1 - predictions  # Negative class
            return proba
            
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
            
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]  # Probability of delay
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        
        return metrics
        
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available."""
        if not self.is_trained or self.feature_importance is None:
            return None
        return self.feature_importance
        
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
            
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'feature_importance': self.feature_importance,
            'training_metrics': self.training_metrics,
            'trained_at': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        self.model_path = filepath
        logger.info(f"Model saved to {filepath}")
        
    def load_model(self, filepath: str) -> None:
        """Load a pre-trained model."""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.model_name = model_data['model_name']
            self.is_trained = model_data['is_trained']
            self.feature_importance = model_data.get('feature_importance')
            self.training_metrics = model_data.get('training_metrics', {})
            self.model_path = filepath
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model from {filepath}: {e}")
            raise


class RandomForestDelayPredictor(DelayPredictionModel):
    """
    Random Forest model for delay prediction.
    """
    
    def __init__(self, **kwargs):
        super().__init__("RandomForest")
        
        # Default hyperparameters
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(kwargs)
        
        self.model = RandomForestClassifier(**default_params)
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, float]:
        """Train the Random Forest model."""
        logger.info(f"Training {self.model_name} model...")
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Get feature importance
        feature_names = X_train.columns.tolist()
        self.feature_importance = dict(zip(feature_names, self.model.feature_importances_))
        
        # Evaluate on training data
        train_metrics = self.evaluate(X_train, y_train)
        self.training_metrics = train_metrics
        
        logger.info(f"Training completed. Accuracy: {train_metrics['accuracy']:.3f}")
        return train_metrics


class XGBoostDelayPredictor(DelayPredictionModel):
    """
    XGBoost model for delay prediction.
    """
    
    def __init__(self, **kwargs):
        super().__init__("XGBoost")
        
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available. Install with: pip install xgboost")
        
        # Default hyperparameters
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(kwargs)
        
        self.model = xgb.XGBClassifier(**default_params)
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, float]:
        """Train the XGBoost model."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available")
            
        logger.info(f"Training {self.model_name} model...")
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Get feature importance
        feature_names = X_train.columns.tolist()
        self.feature_importance = dict(zip(feature_names, self.model.feature_importances_))
        
        # Evaluate on training data
        train_metrics = self.evaluate(X_train, y_train)
        self.training_metrics = train_metrics
        
        logger.info(f"Training completed. Accuracy: {train_metrics['accuracy']:.3f}")
        return train_metrics


class LogisticRegressionDelayPredictor(DelayPredictionModel):
    """
    Logistic Regression model for delay prediction.
    """
    
    def __init__(self, **kwargs):
        super().__init__("LogisticRegression")
        
        default_params = {
            'random_state': 42,
            'max_iter': 1000,
            'C': 1.0
        }
        default_params.update(kwargs)
        
        self.model = LogisticRegression(**default_params)
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, float]:
        """Train the Logistic Regression model."""
        logger.info(f"Training {self.model_name} model...")
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Get feature importance (coefficients)
        feature_names = X_train.columns.tolist()
        coefficients = np.abs(self.model.coef_[0])  # Take absolute values
        self.feature_importance = dict(zip(feature_names, coefficients))
        
        # Evaluate on training data
        train_metrics = self.evaluate(X_train, y_train)
        self.training_metrics = train_metrics
        
        logger.info(f"Training completed. Accuracy: {train_metrics['accuracy']:.3f}")
        return train_metrics


class GradientBoostingDelayPredictor(DelayPredictionModel):
    """
    Gradient Boosting model for delay prediction.
    """
    
    def __init__(self, **kwargs):
        super().__init__("GradientBoosting")
        
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'random_state': 42
        }
        default_params.update(kwargs)
        
        self.model = GradientBoostingClassifier(**default_params)
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, float]:
        """Train the Gradient Boosting model."""
        logger.info(f"Training {self.model_name} model...")
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Get feature importance
        feature_names = X_train.columns.tolist()
        self.feature_importance = dict(zip(feature_names, self.model.feature_importances_))
        
        # Evaluate on training data
        train_metrics = self.evaluate(X_train, y_train)
        self.training_metrics = train_metrics
        
        logger.info(f"Training completed. Accuracy: {train_metrics['accuracy']:.3f}")
        return train_metrics


class EnsembleDelayPredictor:
    """
    Ensemble model combining multiple delay predictors.
    """
    
    def __init__(self, models: List[DelayPredictionModel]):
        self.models = models
        self.weights = None
        self.is_trained = False
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Train all models in the ensemble.
        """
        logger.info("Training ensemble models...")
        
        results = {}
        
        # Train each model
        for model in self.models:
            try:
                train_metrics = model.train(X_train, y_train)
                results[model.model_name] = train_metrics
                logger.info(f"{model.model_name} - Accuracy: {train_metrics['accuracy']:.3f}")
            except Exception as e:
                logger.error(f"Error training {model.model_name}: {e}")
                continue
        
        # Calculate ensemble weights based on validation performance
        if X_val is not None and y_val is not None:
            self._calculate_weights(X_val, y_val)
        else:
            # Equal weights if no validation data
            self.weights = {model.model_name: 1.0 for model in self.models if model.is_trained}
            
        self.is_trained = True
        
        return results
        
    def _calculate_weights(self, X_val: pd.DataFrame, y_val: pd.Series):
        """Calculate ensemble weights based on validation performance."""
        weights = {}
        
        for model in self.models:
            if model.is_trained:
                try:
                    val_metrics = model.evaluate(X_val, y_val)
                    # Use F1 score for weighting (balanced metric)
                    weights[model.model_name] = val_metrics['f1_score']
                except Exception as e:
                    logger.warning(f"Error evaluating {model.model_name}: {e}")
                    weights[model.model_name] = 0.0
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in weights.items()}
        else:
            self.weights = {k: 1.0 / len(weights) for k in weights.keys()}
            
        logger.info(f"Ensemble weights: {self.weights}")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
            
        predictions = []
        total_weight = 0
        
        for model in self.models:
            if model.is_trained and model.model_name in self.weights:
                try:
                    model_pred = model.predict(X)
                    weight = self.weights[model.model_name]
                    predictions.append(model_pred * weight)
                    total_weight += weight
                except Exception as e:
                    logger.warning(f"Error predicting with {model.model_name}: {e}")
                    continue
        
        if not predictions:
            raise ValueError("No models available for prediction")
            
        # Weighted average
        ensemble_pred = np.sum(predictions, axis=0) / total_weight
        
        # Convert to binary predictions
        return (ensemble_pred > 0.5).astype(int)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities using ensemble."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
            
        probabilities = []
        total_weight = 0
        
        for model in self.models:
            if model.is_trained and model.model_name in self.weights:
                try:
                    model_proba = model.predict_proba(X)
                    weight = self.weights[model.model_name]
                    probabilities.append(model_proba * weight)
                    total_weight += weight
                except Exception as e:
                    logger.warning(f"Error predicting probabilities with {model.model_name}: {e}")
                    continue
        
        if not probabilities:
            raise ValueError("No models available for probability prediction")
            
        # Weighted average
        ensemble_proba = np.sum(probabilities, axis=0) / total_weight
        
        return ensemble_proba
        
    def get_model_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get predictions from each individual model."""
        predictions = {}
        
        for model in self.models:
            if model.is_trained:
                try:
                    predictions[model.model_name] = model.predict(X)
                except Exception as e:
                    logger.warning(f"Error getting predictions from {model.model_name}: {e}")
                    
        return predictions


class ModelTrainer:
    """
    Utility class for training and comparing multiple models.
    """
    
    def __init__(self):
        self.trained_models = {}
        
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                        X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict]:
        """
        Train multiple models and compare performance.
        """
        logger.info("Starting comprehensive model training...")
        
        # Define models to train
        models_to_train = [
            RandomForestDelayPredictor(),
            LogisticRegressionDelayPredictor(),
            GradientBoostingDelayPredictor()
        ]
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models_to_train.append(XGBoostDelayPredictor())
        
        results = {}
        
        # Train each model
        for model in models_to_train:
            try:
                logger.info(f"Training {model.model_name}...")
                
                # Train model
                train_metrics = model.train(X_train, y_train)
                
                # Evaluate on test set
                test_metrics = model.evaluate(X_test, y_test)
                
                # Store results
                results[model.model_name] = {
                    'model': model,
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics,
                    'feature_importance': model.get_feature_importance()
                }
                
                self.trained_models[model.model_name] = model
                
                logger.info(f"{model.model_name} - Test Accuracy: {test_metrics['accuracy']:.3f}, "
                          f"F1 Score: {test_metrics['f1_score']:.3f}")
                
            except Exception as e:
                logger.error(f"Error training {model.model_name}: {e}")
                continue
        
        # Train ensemble model
        if len(self.trained_models) > 1:
            try:
                ensemble = EnsembleDelayPredictor(list(self.trained_models.values()))
                
                # Use part of training data for validation to calculate weights
                X_train_sub, X_val, y_train_sub, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
                )
                
                ensemble.train(X_train_sub, y_train_sub, X_val, y_val)
                
                # Evaluate ensemble on test set
                ensemble_pred = ensemble.predict(X_test)
                ensemble_proba = ensemble.predict_proba(X_test)[:, 1]
                
                ensemble_metrics = {
                    'accuracy': accuracy_score(y_test, ensemble_pred),
                    'precision': precision_score(y_test, ensemble_pred, average='weighted'),
                    'recall': recall_score(y_test, ensemble_pred, average='weighted'),
                    'f1_score': f1_score(y_test, ensemble_pred, average='weighted'),
                    'roc_auc': roc_auc_score(y_test, ensemble_proba)
                }
                
                results['Ensemble'] = {
                    'model': ensemble,
                    'test_metrics': ensemble_metrics,
                    'weights': ensemble.weights
                }
                
                logger.info(f"Ensemble - Test Accuracy: {ensemble_metrics['accuracy']:.3f}, "
                          f"F1 Score: {ensemble_metrics['f1_score']:.3f}")
                
            except Exception as e:
                logger.error(f"Error training ensemble: {e}")
        
        return results
        
    def get_best_model(self, results: Dict[str, Dict], metric: str = 'f1_score') -> Tuple[str, Any]:
        """
        Get the best performing model based on specified metric.
        """
        best_score = -1
        best_model_name = None
        best_model = None
        
        for model_name, result in results.items():
            test_metrics = result.get('test_metrics', {})
            score = test_metrics.get(metric, 0)
            
            if score > best_score:
                best_score = score
                best_model_name = model_name
                best_model = result['model']
                
        logger.info(f"Best model: {best_model_name} with {metric}: {best_score:.3f}")
        
        return best_model_name, best_model


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('..')
    from preprocessing.feature_engineering import ShipmentFeatureEngineer, load_and_prepare_data
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Load and prepare data
    data_file = "../data/smart_logistics_dataset.csv"
    df = load_and_prepare_data(data_file)
    
    # Feature engineering
    fe = ShipmentFeatureEngineer()
    X_train, X_test, y_train, y_test = fe.create_train_test_split(df)
    
    # Train models
    trainer = ModelTrainer()
    results = trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    # Display results
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)
    
    for model_name, result in results.items():
        test_metrics = result.get('test_metrics', {})
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {test_metrics.get('accuracy', 0):.3f}")
        print(f"  Precision: {test_metrics.get('precision', 0):.3f}")
        print(f"  Recall:    {test_metrics.get('recall', 0):.3f}")
        print(f"  F1 Score:  {test_metrics.get('f1_score', 0):.3f}")
        print(f"  ROC AUC:   {test_metrics.get('roc_auc', 0):.3f}")
    
    # Get best model
    best_name, best_model = trainer.get_best_model(results)
    print(f"\nBest performing model: {best_name}")
