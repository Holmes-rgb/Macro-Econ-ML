"""
Quick test script to verify the pipeline works.
Uses a subset of models for faster testing.
"""

import sys
import os
sys.path.insert(0, 'src')

from data_loader import load_and_preprocess_data
from models import InflationForecaster
import numpy as np

print("="*60)
print("QUICK PIPELINE TEST")
print("="*60)

# Load data
print("\nLoading data...")
X, y, features = load_and_preprocess_data(data_path='data/fred_qd.csv')

print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")

# Split data
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Train: {len(X_train)} samples")
print(f"Test: {len(X_test)} samples")

# Test ElasticNet (primary model)
print("\n" + "="*60)
print("Testing ElasticNet Model")
print("="*60)

forecaster = InflationForecaster(model_type='elasticnet', random_state=42)
forecaster.train(X_train, y_train, tune_hyperparameters=True, cv_folds=3)

metrics, y_pred = forecaster.evaluate(X_test, y_test)

print(f"\nElasticNet Performance:")
print(f"  RMSE: {metrics['rmse']:.4f}")
print(f"  MAE:  {metrics['mae']:.4f}")
print(f"  R²:   {metrics['r2']:.4f}")
print(f"  MAPE: {metrics['mape']:.2f}%")

# Get feature importance
importance_df = forecaster.get_feature_importance(features)
print(f"\nTop 5 Most Important Features:")
print(importance_df.head(5).to_string(index=False))

print("\n" + "="*60)
print("✓ Pipeline test successful!")
print("="*60)
