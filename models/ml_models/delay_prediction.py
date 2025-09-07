"""
Machine Learning models for shipment delay prediction
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import joblib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from prophet import Prophet
# Simplified imports for demo - using sklearn only
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.optimizers import Adam
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering for delay prediction"""

    def __init__(self):
        self.scalers = {}
        self.encoders = {}

    def create_temporal_features(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Create temporal features from datetime column"""
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])

        # Extract temporal features
        df[f'{date_column}_year'] = df[date_column].dt.year
        df[f'{date_column}_month'] = df[date_column].dt.month
        df[f'{date_column}_day'] = df[date_column].dt.day
        df[f'{date_column}_dayofweek'] = df[date_column].dt.dayofweek
        df[f'{date_column}_hour'] = df[date_column].dt.hour
        df[f'{date_column}_quarter'] = df[date_column].dt.quarter
        df[f'{date_column}_is_weekend'] = (df[date_column].dt.dayofweek >= 5).astype(int)
        df[f'{date_column}_is_month_end'] = df[date_column].dt.is_month_end.astype(int)
        df[f'{date_column}_is_month_start'] = df[date_column].dt.is_month_start.astype(int)

        return df

    def create_route_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create route-based features"""
        df = df.copy()

        # Route complexity (number of stops, distance approximation)
        df['route_complexity'] = df['route'].str.count('-') + 1
        df['is_international'] = (~df['origin_port'].str[:2].eq(df['destination_port'].str[:2])).astype(int)

        # Port features
        df['origin_port_encoded'] = self._encode_categorical('origin_port', df['origin_port'])
        df['destination_port_encoded'] = self._encode_categorical('destination_port', df['destination_port'])

        return df

    def create_cargo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cargo-based features"""
        df = df.copy()

        # Weight categories
        if 'cargo_weight' in df.columns:
            df['cargo_weight_category'] = pd.cut(
                df['cargo_weight'], 
                bins=[0, 1000, 5000, 10000, 50000, float('inf')], 
                labels=['very_light', 'light', 'medium', 'heavy', 'very_heavy']
            )
            df['cargo_weight_log'] = np.log1p(df['cargo_weight'])

        # Cargo type encoding
        if 'cargo_type' in df.columns:
            df['cargo_type_encoded'] = self._encode_categorical('cargo_type', df['cargo_type'])

        # Container type encoding
        if 'container_type' in df.columns:
            df['container_type_encoded'] = self._encode_categorical('container_type', df['container_type'])

        return df

    def create_historical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create historical performance features"""
        df = df.copy()

        # Calculate historical delay patterns
        if 'actual_arrival' in df.columns and 'scheduled_arrival' in df.columns:
            df['actual_delay_hours'] = (
                pd.to_datetime(df['actual_arrival']) - 
                pd.to_datetime(df['scheduled_arrival'])
            ).dt.total_seconds() / 3600

            # Route performance
            route_performance = df.groupby(['origin_port', 'destination_port'])['actual_delay_hours'].agg([
                'mean', 'std', 'median', 'count'
            ]).reset_index()
            route_performance.columns = ['origin_port', 'destination_port', 
                                       'route_avg_delay', 'route_delay_std', 
                                       'route_median_delay', 'route_frequency']

            df = df.merge(route_performance, on=['origin_port', 'destination_port'], how='left')

        return df

    def create_external_features(self, df: pd.DataFrame, weather_data: Dict = None, 
                               port_data: Dict = None) -> pd.DataFrame:
        """Create features from external data sources"""
        df = df.copy()

        # Weather features (if available)
        if weather_data:
            df['weather_severity_score'] = df.apply(
                lambda row: self._calculate_weather_severity(row, weather_data), axis=1
            )

        # Port congestion features (if available)
        if port_data:
            df['origin_port_congestion'] = df['origin_port'].map(
                lambda x: port_data.get(x, {}).get('congestion_level', 0)
            )
            df['destination_port_congestion'] = df['destination_port'].map(
                lambda x: port_data.get(x, {}).get('congestion_level', 0)
            )

        return df

    def _encode_categorical(self, column_name: str, values: pd.Series) -> pd.Series:
        """Encode categorical variables"""
        if column_name not in self.encoders:
            self.encoders[column_name] = LabelEncoder()
            return self.encoders[column_name].fit_transform(values.astype(str))
        else:
            return self.encoders[column_name].transform(values.astype(str))

    def _calculate_weather_severity(self, row: pd.Series, weather_data: Dict) -> float:
        """Calculate weather severity score"""
        # Simplified weather severity calculation
        # In production, this would use actual weather API data
        base_score = 0.5
        if 'scheduled_departure' in row:
            # Add seasonal variations
            month = pd.to_datetime(row['scheduled_departure']).month
            if month in [12, 1, 2]:  # Winter
                base_score += 0.2
            elif month in [6, 7, 8]:  # Summer
                base_score += 0.1
        return base_score

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform features"""
        df = self.create_temporal_features(df, 'scheduled_departure')
        df = self.create_temporal_features(df, 'scheduled_arrival')
        df = self.create_route_features(df)
        df = self.create_cargo_features(df)
        df = self.create_historical_features(df)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted encoders"""
        df = self.create_temporal_features(df, 'scheduled_departure')
        df = self.create_temporal_features(df, 'scheduled_arrival')
        df = self.create_route_features(df)
        df = self.create_cargo_features(df)
        return df


class DelayPredictionModel:
    """Base class for delay prediction models"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.feature_engineer = FeatureEngineer()
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_columns = None

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for training or prediction"""
        # Apply feature engineering
        df_features = self.feature_engineer.fit_transform(df) if not self.is_fitted else self.feature_engineer.transform(df)

        # Select numeric columns for modeling
        numeric_columns = df_features.select_dtypes(include=[np.number]).columns

        if self.feature_columns is None:
            self.feature_columns = numeric_columns.tolist()

        # Fill missing values
        df_features = df_features[self.feature_columns].fillna(0)

        return df_features.values

    def train(self, df: pd.DataFrame, target_column: str = 'actual_delay_hours') -> Dict[str, float]:
        """Train the model"""
        raise NotImplementedError

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        raise NotImplementedError

    def save_model(self, path: Path) -> None:
        """Save model to disk"""
        model_data = {
            'model': self.model,
            'feature_engineer': self.feature_engineer,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_name': self.model_name,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: Path) -> None:
        """Load model from disk"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.feature_engineer = model_data['feature_engineer']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.model_name = model_data['model_name']
        self.is_fitted = model_data['is_fitted']
        logger.info(f"Model loaded from {path}")


class XGBoostDelayPredictor(DelayPredictionModel):
    """XGBoost model for delay prediction"""

    def __init__(self):
        super().__init__("XGBoost")
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

    def train(self, df: pd.DataFrame, target_column: str = 'actual_delay_hours') -> Dict[str, float]:
        """Train XGBoost model"""
        X = self.prepare_features(df)
        y = df[target_column].values

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }

        logger.info(f"XGBoost training completed. Metrics: {metrics}")
        return metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions with XGBoost"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")

        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)

        return np.maximum(predictions, 0)  # Ensure non-negative delays


class RandomForestDelayPredictor(DelayPredictionModel):
    """Random Forest model for delay prediction"""

    def __init__(self):
        super().__init__("RandomForest")
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )

    def train(self, df: pd.DataFrame, target_column: str = 'actual_delay_hours') -> Dict[str, float]:
        """Train Random Forest model"""
        X = self.prepare_features(df)
        y = df[target_column].values

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }

        logger.info(f"Random Forest training completed. Metrics: {metrics}")
        return metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions with Random Forest"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")

        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)

        return np.maximum(predictions, 0)


class LSTMDelayPredictor(DelayPredictionModel):
    """LSTM model for delay prediction"""

    def __init__(self):
        super().__init__("LSTM")
        self.sequence_length = 10
        self.model = None

    def create_sequences(self, data: np.ndarray, target: np.ndarray, sequence_length: int):
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(target[i + sequence_length])
        return np.array(X), np.array(y)

    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """Build LSTM model"""
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

    def train(self, df: pd.DataFrame, target_column: str = 'actual_delay_hours') -> Dict[str, float]:
        """Train LSTM model"""
        X = self.prepare_features(df)
        y = df[target_column].values

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled, y, self.sequence_length)

        # Split data
        split_idx = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

        # Build model
        self.build_model((self.sequence_length, X_scaled.shape[1]))

        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=0
        )

        self.is_fitted = True

        # Evaluate
        y_pred = self.model.predict(X_test)
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }

        logger.info(f"LSTM training completed. Metrics: {metrics}")
        return metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions with LSTM"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")

        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)

        # For prediction, we need sequences
        # If we don't have enough history, use the last available sequence repeated
        if len(X_scaled) < self.sequence_length:
            # Pad with zeros or repeat last row
            padding = np.zeros((self.sequence_length - len(X_scaled), X_scaled.shape[1]))
            X_scaled = np.vstack([padding, X_scaled])

        X_seq = []
        for i in range(len(X_scaled) - self.sequence_length + 1):
            X_seq.append(X_scaled[i:(i + self.sequence_length)])
        X_seq = np.array(X_seq)

        predictions = self.model.predict(X_seq)
        return np.maximum(predictions.flatten(), 0)


class ProphetDelayPredictor:
    """Prophet model for time series delay prediction"""

    def __init__(self):
        self.model_name = "Prophet"
        self.models = {}  # Store models per route
        self.is_fitted = False

    def prepare_prophet_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for Prophet"""
        prophet_df = df.copy()
        prophet_df['ds'] = pd.to_datetime(prophet_df['scheduled_departure'])
        prophet_df['y'] = prophet_df['actual_delay_hours']

        # Add regressors
        if 'cargo_weight' in df.columns:
            prophet_df['cargo_weight'] = df['cargo_weight'].fillna(0)
        if 'is_priority' in df.columns:
            prophet_df['is_priority'] = df['is_priority'].astype(int)

        return prophet_df[['ds', 'y'] + [col for col in ['cargo_weight', 'is_priority'] if col in prophet_df.columns]]

    def train(self, df: pd.DataFrame, target_column: str = 'actual_delay_hours') -> Dict[str, float]:
        """Train Prophet models per route"""
        metrics = {}

        for route in df['route'].unique():
            if pd.isna(route):
                continue

            route_data = df[df['route'] == route].copy()
            if len(route_data) < 10:  # Need minimum data points
                continue

            prophet_data = self.prepare_prophet_data(route_data)

            # Create and train Prophet model
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )

            # Add regressors if available
            if 'cargo_weight' in prophet_data.columns:
                model.add_regressor('cargo_weight')
            if 'is_priority' in prophet_data.columns:
                model.add_regressor('is_priority')

            try:
                model.fit(prophet_data)
                self.models[route] = model

                # Evaluate
                future = model.make_future_dataframe(periods=30)  # 30 days ahead
                if 'cargo_weight' in prophet_data.columns:
                    future['cargo_weight'] = prophet_data['cargo_weight'].mean()
                if 'is_priority' in prophet_data.columns:
                    future['is_priority'] = 0

                forecast = model.predict(future)

                # Calculate metrics on training data
                y_true = prophet_data['y'].values
                y_pred = forecast['yhat'][:len(y_true)].values

                route_metrics = {
                    'mae': mean_absolute_error(y_true, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                    'r2': r2_score(y_true, y_pred)
                }
                metrics[route] = route_metrics

            except Exception as e:
                logger.warning(f"Failed to train Prophet model for route {route}: {e}")
                continue

        self.is_fitted = True
        logger.info(f"Prophet training completed for {len(self.models)} routes")
        return metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions with Prophet"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")

        predictions = []

        for _, row in df.iterrows():
            route = row.get('route')
            if route in self.models:
                model = self.models[route]

                # Create future dataframe for this prediction
                future_data = pd.DataFrame({
                    'ds': [pd.to_datetime(row['scheduled_departure'])]
                })

                # Add regressors
                if 'cargo_weight' in row:
                    future_data['cargo_weight'] = row['cargo_weight'] if pd.notna(row['cargo_weight']) else 0
                if 'is_priority' in row:
                    future_data['is_priority'] = int(row['is_priority']) if pd.notna(row['is_priority']) else 0

                try:
                    forecast = model.predict(future_data)
                    prediction = max(forecast['yhat'].iloc[0], 0)
                    predictions.append(prediction)
                except Exception as e:
                    logger.warning(f"Prophet prediction failed for route {route}: {e}")
                    predictions.append(0)
            else:
                # No model for this route, use average delay
                predictions.append(0)

        return np.array(predictions)


class EnsembleDelayPredictor:
    """Ensemble model combining multiple predictors"""

    def __init__(self):
        self.models = {
            'xgboost': XGBoostDelayPredictor(),
            'random_forest': RandomForestDelayPredictor(),
            'lstm': LSTMDelayPredictor()
        }
        self.weights = {'xgboost': 0.4, 'random_forest': 0.3, 'lstm': 0.3}
        self.is_fitted = False

    def train(self, df: pd.DataFrame, target_column: str = 'actual_delay_hours') -> Dict[str, Dict[str, float]]:
        """Train all models in ensemble"""
        all_metrics = {}

        for name, model in self.models.items():
            try:
                metrics = model.train(df, target_column)
                all_metrics[name] = metrics
                logger.info(f"Trained {name} successfully")
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
                all_metrics[name] = {'error': str(e)}

        self.is_fitted = True
        return all_metrics

    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Make ensemble predictions"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be trained before making predictions")

        predictions = {}
        valid_predictions = []

        for name, model in self.models.items():
            try:
                pred = model.predict(df)
                predictions[name] = pred
                valid_predictions.append(pred * self.weights[name])
            except Exception as e:
                logger.warning(f"Prediction failed for {name}: {e}")
                predictions[name] = np.zeros(len(df))

        # Ensemble prediction
        if valid_predictions:
            ensemble_pred = np.sum(valid_predictions, axis=0)
        else:
            ensemble_pred = np.zeros(len(df))

        return {
            'ensemble': ensemble_pred,
            'individual': predictions,
            'confidence': self._calculate_confidence(predictions)
        }

    def _calculate_confidence(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate prediction confidence based on model agreement"""
        pred_array = np.array(list(predictions.values()))
        if len(pred_array) > 1:
            # Use coefficient of variation as confidence measure
            mean_pred = np.mean(pred_array, axis=0)
            std_pred = np.std(pred_array, axis=0)
            cv = np.where(mean_pred != 0, std_pred / np.abs(mean_pred), 1)
            confidence = 1 / (1 + cv)  # Higher confidence when predictions agree
            return np.clip(confidence, 0, 1)
        else:
            return np.ones(pred_array.shape[1]) * 0.5

    def save_ensemble(self, path: Path) -> None:
        """Save ensemble models"""
        ensemble_data = {
            'weights': self.weights,
            'is_fitted': self.is_fitted
        }

        # Save individual models
        for name, model in self.models.items():
            model_path = path / f"{name}_model.pkl"
            model.save_model(model_path)

        # Save ensemble metadata
        joblib.dump(ensemble_data, path / "ensemble_metadata.pkl")
        logger.info(f"Ensemble saved to {path}")

    def load_ensemble(self, path: Path) -> None:
        """Load ensemble models"""
        # Load ensemble metadata
        ensemble_data = joblib.load(path / "ensemble_metadata.pkl")
        self.weights = ensemble_data['weights']
        self.is_fitted = ensemble_data['is_fitted']

        # Load individual models
        for name, model in self.models.items():
            model_path = path / f"{name}_model.pkl"
            if model_path.exists():
                model.load_model(model_path)

        logger.info(f"Ensemble loaded from {path}")
