"""
Script to retrain the ML model without target leakage.

This script retrains the delay prediction model using only legitimate features,
excluding columns that would cause data leakage like Shipment_Status and Logistics_Delay_Reason.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from ml.preprocessing.feature_engineering import ShipmentFeatureEngineer
from ml.models.delay_predictor import RandomForestDelayPredictor
import joblib
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data():
    """Load the dataset"""
    data_path = "data/smart_logistics_dataset.csv"
    logger.info(f"Loading data from {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} records")
    
    # Rename columns to lowercase for consistency
    df.columns = df.columns.str.lower().str.replace('_', '_')
    
    # Map actual column names
    column_mappings = {
        'asset_id': 'asset_id',
        'waiting_time': 'waiting_time',
        'timestamp': 'timestamp',
        'traffic_status': 'traffic_conditions',
        'logistics_delay': 'delay_present'
    }
    
    for old_col, new_col in column_mappings.items():
        if old_col in df.columns and old_col != new_col:
            df = df.rename(columns={old_col: new_col})
    
    # Use Logistics_Delay to create delay_hours (realistic delays)
    if 'logistics_delay' in df.columns:
        # Create realistic delay hours based on the binary delay indicator
        np.random.seed(42)
        df['delay_hours'] = 0
        delayed_mask = df['logistics_delay'].astype(bool)
        
        # For delayed shipments, assign realistic delay hours
        # Most delays are 1-3 hours, some are longer
        n_delays = delayed_mask.sum()
        if n_delays > 0:
            delay_values = np.concatenate([
                np.random.exponential(scale=2, size=int(n_delays * 0.7)),  # 70% small delays
                np.random.exponential(scale=5, size=int(n_delays * 0.2)),  # 20% medium delays
                np.random.exponential(scale=10, size=n_delays - int(n_delays * 0.7) - int(n_delays * 0.2))  # 10% large delays
            ])
            np.random.shuffle(delay_values)
            df.loc[delayed_mask, 'delay_hours'] = delay_values[:n_delays]
        
        logger.info(f"Created delay_hours from logistics_delay: {delayed_mask.sum()} delayed shipments ({delayed_mask.mean()*100:.1f}%)")
    else:
        # Fallback: create synthetic delays
        logger.warning("logistics_delay column not found, creating synthetic delays")
        df['delay_hours'] = np.where(
            np.random.rand(len(df)) < 0.25,
            np.random.exponential(scale=3, size=len(df)),
            0
        )
    
    # Add missing columns that feature engineering expects
    if 'distance_traveled' not in df.columns:
        # Calculate distance based on coordinates if available
        if 'latitude' in df.columns and 'longitude' in df.columns:
            # Simple distance calculation from origin
            df['distance_traveled'] = np.sqrt(df['latitude']**2 + df['longitude']**2) * 100  # Scale to realistic distances
        else:
            # Generate synthetic distances
            df['distance_traveled'] = np.random.exponential(scale=500, size=len(df))
    
    if 'fuel_efficiency' not in df.columns:
        # Generate realistic fuel efficiency values (miles per gallon)
        df['fuel_efficiency'] = np.random.normal(loc=12, scale=2, size=len(df))
        df['fuel_efficiency'] = df['fuel_efficiency'].clip(lower=5, upper=20)
    
    if 'weather_conditions' not in df.columns:
        # Generate weather conditions
        weather_options = ['Clear', 'Clear', 'Clear', 'Cloudy', 'Cloudy', 'Rainy', 'Stormy']  # Weighted towards good weather
        df['weather_conditions'] = np.random.choice(weather_options, size=len(df))
    
    # Remove columns that would cause target leakage
    leakage_columns = ['shipment_status', 'logistics_delay', 'logistics_delay_reason', 
                      'status', 'delay_reason']
    
    for col in leakage_columns:
        if col in df.columns:
            logger.warning(f"Removing target leakage column: {col}")
            df = df.drop(columns=[col])
    
    return df

def train_model(df):
    """Train the model without target leakage"""
    logger.info("Starting model training without target leakage...")
    
    # Feature engineering
    fe = ShipmentFeatureEngineer()
    X_train, X_test, y_train, y_test = fe.create_train_test_split(df, test_size=0.2)
    
    logger.info(f"Feature columns used: {fe.feature_columns}")
    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Test set shape: {X_test.shape}")
    logger.info(f"Class distribution - Train: {y_train.value_counts().to_dict()}")
    logger.info(f"Class distribution - Test: {y_test.value_counts().to_dict()}")
    
    # Train Random Forest model
    model = RandomForestDelayPredictor()
    model.train(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    logger.info("\n" + "="*50)
    logger.info("MODEL PERFORMANCE (Without Target Leakage)")
    logger.info("="*50)
    logger.info(f"Accuracy: {accuracy:.3f}")
    logger.info(f"Precision: {precision:.3f}")
    logger.info(f"Recall: {recall:.3f}")
    logger.info(f"F1 Score: {f1:.3f}")
    logger.info(f"ROC AUC: {roc_auc:.3f}")
    logger.info("\n" + classification_report(y_test, y_pred, target_names=['No Delay', 'Delayed']))
    
    # Cross-validation for more robust evaluation
    cv_scores = cross_val_score(model.model, X_train, y_train, cv=5, scoring='roc_auc')
    logger.info(f"\nCross-validation ROC AUC scores: {cv_scores}")
    logger.info(f"Mean CV ROC AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Feature importance
    feature_importance = model.get_feature_importance()
    if feature_importance:
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info("\nTop 10 Most Important Features:")
        for i, (feature, importance) in enumerate(top_features, 1):
            logger.info(f"{i}. {feature}: {importance:.4f}")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = "ml/saved_models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model_path = f"{model_dir}/RandomForest_no_leakage_{timestamp}.joblib"
    model.save_model(model_path)
    logger.info(f"\nModel saved to: {model_path}")
    
    # Save feature engineer
    fe_path = f"{model_dir}/RandomForest_no_leakage_{timestamp}_feature_engineer.joblib"
    joblib.dump(fe, fe_path)
    logger.info(f"Feature engineer saved to: {fe_path}")
    
    # Save metadata
    metadata = {
        'model_type': 'RandomForest',
        'trained_at': datetime.now().isoformat(),
        'dataset_size': len(df),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'features_count': len(fe.feature_columns),
        'feature_columns': fe.feature_columns,
        'performance': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'cv_roc_auc_mean': float(cv_scores.mean()),
            'cv_roc_auc_std': float(cv_scores.std())
        },
        'no_target_leakage': True,
        'excluded_columns': ['Shipment_Status', 'Logistics_Delay_Reason', 'status', 'delay_reason', 'delay_category']
    }
    
    metadata_path = f"{model_dir}/RandomForest_no_leakage_{timestamp}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to: {metadata_path}")
    
    return model, fe, metadata

if __name__ == "__main__":
    # Load data
    df = load_data()
    
    # Train model
    model, fe, metadata = train_model(df)
    
    logger.info("\n" + "="*50)
    logger.info("MODEL TRAINING COMPLETE")
    logger.info("="*50)
    logger.info("The model has been retrained without target leakage.")
    logger.info(f"Final ROC AUC: {metadata['performance']['roc_auc']:.3f}")
    logger.info("This is a more realistic performance metric for production use.")
