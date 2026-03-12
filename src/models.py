"""
Inflation forecasting models using scikit-learn.
Implements ElasticNet regression for economic forecasting.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings


class InflationForecaster:
    """
    A class for training and evaluating inflation forecasting models using ElasticNet.
    """

    def __init__(self, random_state=42):
        """
        Initialize the ElasticNet forecaster.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.best_params = None

    def _get_model(self):
        """Return an ElasticNet model."""
        return ElasticNet(random_state=self.random_state, max_iter=10000)

    def _get_param_grid(self):
        """Return hyperparameter grid for ElasticNet."""
        return {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
        }
    
    def train(self, X_train, y_train, tune_hyperparameters=True, cv_folds=5):
        """
        Train the model with optional hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training target
            tune_hyperparameters: Whether to perform grid search
            cv_folds: Number of cross-validation folds
        
        Returns:
            self
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if tune_hyperparameters:
            print("Tuning ElasticNet hyperparameters...")
            base_model = self._get_model()
            param_grid = self._get_param_grid()
            
            # Use TimeSeriesSplit for time series data
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train_scaled, y_train)
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            
            print(f"Best parameters: {self.best_params}")
            print(f"Best CV score (RMSE): {np.sqrt(-grid_search.best_score_):.4f}")
        else:
            print("Training ElasticNet with default parameters...")
            self.model = self._get_model()
            self.model.fit(X_train_scaled, y_train)
        
        return self
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict on
        
        Returns:
            predictions: Array of predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test target
        
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100  # Add epsilon to avoid division by zero
        }
        
        return metrics, y_pred
    
    def get_feature_importance(self, feature_names):
        """
        Get feature importance for the model.
        
        Args:
            feature_names: List of feature names
        
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        if hasattr(self.model, 'coef_'):
            # Linear models
            importance = np.abs(self.model.coef_)
        elif hasattr(self.model, 'feature_importances_'):
            # Tree-based models
            importance = self.model.feature_importances_
        else:
            return None
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df


if __name__ == "__main__":
    from data_loader import get_latest_vintage, build_dataset_with_fred_target
    import sys
    import os

    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data'
    api_key = os.environ.get('FRED_API_KEY', '')
    train_file = get_latest_vintage(data_dir)
    X_train, y_train, X_val, y_val, X_test, y_test, features = \
        build_dataset_with_fred_target(train_file, api_key)

    forecaster = InflationForecaster()
    forecaster.train(X_train, y_train, tune_hyperparameters=True, cv_folds=5)

    metrics, _ = forecaster.evaluate(X_test, y_test)
    print("\n" + "="*60)
    print("ELASTICNET RESULTS")
    print("="*60)
    print(f"  Best params : {forecaster.best_params}")
    print(f"  RMSE : {metrics['rmse']:.4f}")
    print(f"  MAE  : {metrics['mae']:.4f}")
    print(f"  R²   : {metrics['r2']:.4f}")
    print(f"  MAPE : {metrics['mape']:.2f}%")
