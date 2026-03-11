"""
Inflation forecasting models using scikit-learn.
Implements ElasticNet and other regression models for economic forecasting.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class InflationForecaster:
    """
    A class for training and evaluating inflation forecasting models.
    """
    
    def __init__(self, model_type='elasticnet', random_state=42):
        """
        Initialize the forecaster with a specific model type.
        
        Args:
            model_type: Type of model to use ('elasticnet', 'ridge', 'lasso', 
                       'random_forest', 'gradient_boosting')
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.best_params = None
        
    def _get_model(self):
        """Get the model based on model_type."""
        if self.model_type == 'elasticnet':
            return ElasticNet(random_state=self.random_state, max_iter=10000)
        elif self.model_type == 'ridge':
            return Ridge(max_iter=10000)
        elif self.model_type == 'lasso':
            return Lasso(max_iter=10000)
        elif self.model_type == 'random_forest':
            return RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingRegressor(random_state=self.random_state)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _get_param_grid(self):
        """Get parameter grid for hyperparameter tuning."""
        if self.model_type == 'elasticnet':
            return {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            }
        elif self.model_type == 'ridge':
            return {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
            }
        elif self.model_type == 'lasso':
            return {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
            }
        elif self.model_type == 'random_forest':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            }
        elif self.model_type == 'gradient_boosting':
            return {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        else:
            return {}
    
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
            print(f"Tuning hyperparameters for {self.model_type}...")
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
            print(f"Training {self.model_type} with default parameters...")
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


def compare_models(X, y, test_size=0.2, random_state=42):
    """
    Compare multiple models on the same dataset.
    
    Args:
        X: Feature matrix
        y: Target variable
        test_size: Proportion of data for testing
        random_state: Random seed
    
    Returns:
        results: DataFrame with model comparison results
    """
    # Split data (preserve time order for time series)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Models to compare
    model_types = ['elasticnet', 'ridge', 'lasso', 'random_forest', 'gradient_boosting']
    
    results = []
    predictions = {}
    
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()} model...")
        print('='*60)
        
        try:
            forecaster = InflationForecaster(model_type=model_type, random_state=random_state)
            forecaster.train(X_train, y_train, tune_hyperparameters=True, cv_folds=3)
            
            metrics, y_pred = forecaster.evaluate(X_test, y_test)
            predictions[model_type] = y_pred
            
            print(f"\nTest Set Performance:")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  MAE:  {metrics['mae']:.4f}")
            print(f"  R²:   {metrics['r2']:.4f}")
            print(f"  MAPE: {metrics['mape']:.2f}%")
            
            results.append({
                'model': model_type,
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'r2': metrics['r2'],
                'mape': metrics['mape'],
                'best_params': str(forecaster.best_params)
            })
        except Exception as e:
            print(f"Error training {model_type}: {e}")
            continue
    
    results_df = pd.DataFrame(results).sort_values('rmse')
    
    return results_df, predictions, X_test, y_test


if __name__ == "__main__":
    from data_loader import get_latest_vintage, build_dataset
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data'
    train_file = get_latest_vintage(data_dir)
    X, y, _, _, features = build_dataset(train_file, train_file)
    
    # Compare models
    results, predictions, X_test, y_test = compare_models(X, y)
    
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)
    print(results.to_string(index=False))
    
    print(f"\nBest model: {results.iloc[0]['model']} (RMSE: {results.iloc[0]['rmse']:.4f})")
