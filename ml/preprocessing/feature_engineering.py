"""
Feature Engineering Pipeline for Shipment Delay Prediction

This module handles data preprocessing and feature engineering for ML models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class ShipmentFeatureEngineer:
    """
    Feature engineering pipeline for shipment delay prediction.
    
    Transforms raw shipment data into ML-ready features.
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        self.is_fitted = False
        
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from timestamps.
        """
        df = df.copy()
        
        # Ensure datetime conversion
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Extract time components
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            df['quarter'] = df['timestamp'].dt.quarter
            
            # Create cyclical features for time
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            
        return df
    
    def create_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create location-based features from coordinates.
        """
        df = df.copy()
        
        if 'latitude' in df.columns and 'longitude' in df.columns:
            # Distance from origin (assuming (0,0) as reference)
            df['distance_from_origin'] = np.sqrt(df['latitude']**2 + df['longitude']**2)
            
            # Coordinate regions (simple binning)
            df['lat_region'] = pd.cut(df['latitude'], bins=5, labels=['lat_1', 'lat_2', 'lat_3', 'lat_4', 'lat_5'])
            df['lon_region'] = pd.cut(df['longitude'], bins=5, labels=['lon_1', 'lon_2', 'lon_3', 'lon_4', 'lon_5'])
            
            # Convert categorical regions to strings for encoding
            df['lat_region'] = df['lat_region'].astype(str)
            df['lon_region'] = df['lon_region'].astype(str)
            
        return df
    
    def create_delay_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create delay-related features and target variables.
        
        NOTE: Only creates the target variable 'is_delayed'.
        Other delay-based features would cause target leakage.
        """
        df = df.copy()
        
        # Binary delay indicator (TARGET VARIABLE)
        df['is_delayed'] = (df['delay_hours'] > 0).astype(int)
        
        # DO NOT create delay_category as it's derived from target and causes leakage
        
        # Risk level based on multiple factors
        def calculate_risk_level(row):
            risk_score = 0
            
            # Traffic conditions
            if row.get('traffic_conditions') == 'Heavy':
                risk_score += 3
            elif row.get('traffic_conditions') == 'Moderate':
                risk_score += 1
                
            # Weather conditions
            if row.get('weather_conditions') == 'Stormy':
                risk_score += 3
            elif row.get('weather_conditions') in ['Rainy', 'Snowy']:
                risk_score += 2
            elif row.get('weather_conditions') == 'Cloudy':
                risk_score += 1
                
            # Fuel efficiency (lower efficiency might indicate issues)
            if row.get('fuel_efficiency', 0) < 8:
                risk_score += 2
            elif row.get('fuel_efficiency', 0) < 10:
                risk_score += 1
                
            # Distance traveled (longer distances = higher risk)
            if row.get('distance_traveled', 0) > 1000:
                risk_score += 2
            elif row.get('distance_traveled', 0) > 500:
                risk_score += 1
                
            return min(risk_score, 10)  # Cap at 10
        
        df['risk_score'] = df.apply(calculate_risk_level, axis=1)
        
        return df
    
    def create_asset_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create asset-specific features.
        """
        df = df.copy()
        
        if 'asset_id' in df.columns:
            # Only calculate stats if we have the required columns
            agg_dict = {}
            if 'delay_hours' in df.columns:
                agg_dict['delay_hours'] = ['mean', 'std', 'count']
            if 'fuel_efficiency' in df.columns:
                agg_dict['fuel_efficiency'] = ['mean']
            if 'distance_traveled' in df.columns:
                agg_dict['distance_traveled'] = ['mean']
            
            if agg_dict:
                # Asset performance history (simplified)
                asset_stats = df.groupby('asset_id').agg(agg_dict).round(2)
                
                # Flatten column names
                asset_stats.columns = ['_'.join(col).strip() for col in asset_stats.columns.values]
                asset_stats = asset_stats.add_prefix('asset_')
                
                # Merge back to original dataframe
                df = df.merge(asset_stats, left_on='asset_id', right_index=True, how='left')
            
            # Encode asset_id as a categorical feature
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df['asset_id_encoded'] = le.fit_transform(df['asset_id'].fillna('unknown'))
            
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Encode categorical variables.
        
        IMPORTANT: Excludes 'status' and 'delay_reason' to avoid target leakage.
        These fields directly indicate if there was a delay, making prediction trivial.
        """
        df = df.copy()
        
        # Exclude status and delay_reason to prevent target leakage
        # These columns contain information that's only available AFTER we know if there's a delay
        categorical_columns = [
            'traffic_conditions', 'weather_conditions', 
            'lat_region', 'lon_region'
            # Removed: 'status', 'delay_reason', 'delay_category' - these cause target leakage
        ]
        
        for col in categorical_columns:
            if col in df.columns:
                if fit:
                    if col not in self.encoders:
                        self.encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col].fillna('unknown'))
                else:
                    if col in self.encoders:
                        # Handle unseen categories
                        def safe_transform(x):
                            try:
                                return self.encoders[col].transform([x])[0]
                            except ValueError:
                                # Return most frequent class for unseen categories
                                return 0
                        
                        df[f'{col}_encoded'] = df[col].fillna('unknown').apply(safe_transform)
        
        return df
    
    def scale_numerical_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Scale numerical features.
        """
        df = df.copy()
        
        numerical_columns = [
            'latitude', 'longitude', 'distance_traveled', 'fuel_efficiency',
            'waiting_time', 'distance_from_origin', 'risk_score',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos'
        ]
        
        # Include asset features if they exist (only numeric ones)
        asset_features = [col for col in df.columns if col.startswith('asset_') and col.endswith(('_mean', '_std', '_count'))]
        numerical_columns.extend(asset_features)
        
        # Only scale columns that exist in the dataframe
        numerical_columns = [col for col in numerical_columns if col in df.columns]
        
        if numerical_columns:
            if fit:
                self.scalers['numerical'] = StandardScaler()
                df[numerical_columns] = self.scalers['numerical'].fit_transform(df[numerical_columns].fillna(0))
            else:
                if 'numerical' in self.scalers:
                    df[numerical_columns] = self.scalers['numerical'].transform(df[numerical_columns].fillna(0))
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        """
        logger.info(f"Starting feature engineering for {len(df)} records")
        
        # Create all feature types
        df = self.create_time_features(df)
        df = self.create_location_features(df)
        df = self.create_delay_features(df)
        df = self.create_asset_features(df)
        
        # Encode and scale
        df = self.encode_categorical_features(df, fit=fit)
        df = self.scale_numerical_features(df, fit=fit)
        
        if fit:
            # Define final feature columns for consistency
            feature_cols = []
            
            # Numerical features
            numerical_features = [
                'latitude', 'longitude', 'distance_traveled', 'fuel_efficiency',
                'waiting_time', 'distance_from_origin', 'risk_score',
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
                'hour', 'day_of_week', 'month', 'quarter', 'asset_id_encoded'
            ]
            feature_cols.extend([col for col in numerical_features if col in df.columns])
            
            # Encoded categorical features
            categorical_encoded = [col for col in df.columns if col.endswith('_encoded')]
            feature_cols.extend(categorical_encoded)
            
            # Asset features (only numeric ones)
            asset_features = [col for col in df.columns if col.startswith('asset_') and (col.endswith(('_mean', '_std', '_count')) or col == 'asset_id_encoded')]
            feature_cols.extend(asset_features)
            
            self.feature_columns = feature_cols
            self.is_fitted = True
        
        logger.info(f"Feature engineering completed. Generated {len(self.feature_columns)} features")
        return df
    
    def get_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract features and target variable for ML training.
        """
        if not self.is_fitted:
            raise ValueError("Feature engineer must be fitted first")
        
        # Ensure all feature columns exist
        available_features = [col for col in self.feature_columns if col in df.columns]
        
        if len(available_features) < len(self.feature_columns):
            missing = set(self.feature_columns) - set(available_features)
            logger.warning(f"Missing feature columns: {missing}")
        
        X = df[available_features].fillna(0)
        y = df['is_delayed'] if 'is_delayed' in df.columns else None
        
        return X, y
    
    def create_train_test_split(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple:
        """
        Create train/test split with proper feature engineering.
        """
        # Fit on full dataset for feature engineering
        df_processed = self.prepare_features(df, fit=True)
        X, y = self.get_features_and_target(df_processed)
        
        if y is None:
            raise ValueError("Target variable 'is_delayed' not found")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"Train/test split created: {len(X_train)} train, {len(X_test)} test samples")
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_importance_names(self) -> List[str]:
        """
        Get feature names for importance analysis.
        """
        return self.feature_columns if self.is_fitted else []


def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """
    Load and prepare shipment data for ML pipeline.
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} records from {file_path}")
        
        # Basic data validation and cleaning
        if 'delay_hours' not in df.columns:
            df['delay_hours'] = np.random.exponential(scale=2, size=len(df))
            # Set 70% of rows to have no delay
            no_delay_indices = np.random.choice(df.index, size=int(0.7 * len(df)), replace=False)
            df.loc[no_delay_indices, 'delay_hours'] = 0
            logger.warning("delay_hours column not found, generated synthetic values")
        
        # Ensure required columns exist with defaults
        required_columns = {
            'asset_id': 'Unknown',
            'status': 'In Transit', 
            'traffic_conditions': 'Normal',
            'weather_conditions': 'Clear',
            'delay_reason': 'None'
        }
        
        for col, default_val in required_columns.items():
            if col not in df.columns:
                df[col] = default_val
                logger.warning(f"Column {col} not found, filled with default value: {default_val}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('..')
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Load data
    data_file = "../data/smart_logistics_dataset.csv"
    df = load_and_prepare_data(data_file)
    
    # Create feature engineer
    fe = ShipmentFeatureEngineer()
    
    # Create train/test split
    X_train, X_test, y_train, y_test = fe.create_train_test_split(df)
    
    print(f"Features created: {len(fe.feature_columns)}")
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"Target distribution - Train: {y_train.value_counts().to_dict()}")
    print(f"Feature columns: {fe.feature_columns[:10]}...")  # Show first 10
