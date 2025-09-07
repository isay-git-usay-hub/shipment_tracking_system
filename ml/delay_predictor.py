"""
Machine Learning Service for Delay Prediction
Uses RandomForest classifier for accurate delay predictions
"""
import logging
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logger = logging.getLogger(__name__)


class DelayPredictor:
    """RandomForest-based delay prediction model"""
    
    def __init__(self, model_path: str = "models/delay_predictor.pkl"):
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.is_trained = False
        
        # Try to load existing model
        self.load_model()
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for training or prediction"""
        df = df.copy()
        
        # Convert timestamp to datetime features
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df['hour'] = df['Timestamp'].dt.hour
            df['day_of_week'] = df['Timestamp'].dt.dayofweek
            df['month'] = df['Timestamp'].dt.month
        
        # Encode categorical variables
        categorical_cols = ['Asset_ID', 'Shipment_Status', 'Traffic_Status', 'Logistics_Delay_Reason']
        
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    # Handle unseen categories
                    df[f'{col}_encoded'] = df[col].apply(
                        lambda x: self.label_encoders[col].transform([str(x)])[0] 
                        if str(x) in self.label_encoders[col].classes_ 
                        else -1
                    )
        
        # Select numerical features
        feature_cols = [
            'Latitude', 'Longitude', 'Inventory_Level', 'Temperature', 'Humidity',
            'Waiting_Time', 'User_Transaction_Amount', 'User_Purchase_Frequency',
            'Asset_Utilization', 'Demand_Forecast', 'hour', 'day_of_week', 'month'
        ]
        
        # Add encoded categorical features
        for col in categorical_cols:
            if f'{col}_encoded' in df.columns:
                feature_cols.append(f'{col}_encoded')
        
        # Filter to existing columns
        self.feature_columns = [col for col in feature_cols if col in df.columns]
        
        return df[self.feature_columns]
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, float]:
        """Train the RandomForest model"""
        logger.info("Training RandomForest delay prediction model...")
        
        # Prepare features
        X = self.prepare_features(df)
        y = df['Logistics_Delay'].astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train RandomForest
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        self.is_trained = True
        
        # Save model
        self.save_model()
        
        logger.info(f"Model trained successfully. Accuracy: {metrics['accuracy']:.3f}")
        return metrics
    
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict delay probability for shipments"""
        if not self.is_trained:
            logger.warning("Model not trained. Using default predictions.")
            # Return heuristic predictions as fallback
            return self._heuristic_predictions(df)
        
        try:
            X = self.prepare_features(df)
            X_scaled = self.scaler.transform(X)
            
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)[:, 1]
            
            return predictions, probabilities
        except Exception as e:
            logger.error(f"Prediction error: {e}. Using heuristic fallback.")
            return self._heuristic_predictions(df)
    
    def _heuristic_predictions(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback heuristic predictions"""
        probabilities = []
        
        for _, row in df.iterrows():
            prob = 0.3  # Base probability
            
            # Adjust based on factors
            if 'Traffic_Status' in row and row['Traffic_Status'] == 'Heavy':
                prob += 0.3
            if 'Waiting_Time' in row and row['Waiting_Time'] > 30:
                prob += 0.2
            if 'Logistics_Delay_Reason' in row and row['Logistics_Delay_Reason'] != 'None':
                prob += 0.2
            
            probabilities.append(min(prob, 1.0))
        
        probabilities = np.array(probabilities)
        predictions = (probabilities > 0.5).astype(int)
        
        return predictions, probabilities
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if not self.is_trained or self.model is None:
            return {}
        
        importance_dict = {}
        for feature, importance in zip(self.feature_columns, self.model.feature_importances_):
            importance_dict[feature] = float(importance)
        
        # Sort by importance
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def save_model(self):
        """Save trained model to disk"""
        if self.is_trained:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_columns': self.feature_columns,
                'trained_at': datetime.now().isoformat()
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {self.model_path}")
    
    def load_model(self) -> bool:
        """Load model from disk"""
        if self.model_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.label_encoders = model_data['label_encoders']
                self.feature_columns = model_data['feature_columns']
                self.is_trained = True
                
                logger.info(f"Model loaded from {self.model_path}")
                return True
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                return False
        return False


# Global instance
_delay_predictor = None

def get_delay_predictor() -> DelayPredictor:
    """Get or create the delay predictor singleton"""
    global _delay_predictor
    if _delay_predictor is None:
        _delay_predictor = DelayPredictor()
    return _delay_predictor


if __name__ == "__main__":
    # Test the model
    import pandas as pd
    
    # Load data
    df = pd.read_csv("data/smart_logistics_dataset.csv")
    print(f"Loaded {len(df)} records")
    
    # Initialize and train model
    predictor = get_delay_predictor()
    metrics = predictor.train(df)
    
    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    # Feature importance
    importance = predictor.get_feature_importance()
    print("\nTop 5 Important Features:")
    for feature, score in list(importance.items())[:5]:
        print(f"  {feature}: {score:.3f}")
    
    # Test prediction on sample
    sample = df.head(5)
    predictions, probabilities = predictor.predict(sample)
    
    print("\nSample Predictions:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        actual = sample.iloc[i]['Logistics_Delay']
        print(f"  Record {i+1}: Predicted={pred}, Probability={prob:.3f}, Actual={actual}")
