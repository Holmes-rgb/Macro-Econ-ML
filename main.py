"""
Main script for inflation forecasting using ElasticNet.

Loads FRED-MD vintage data, trains an ElasticNet model, and reports metrics.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import get_latest_vintage, build_dataset_with_fred_target
from models import InflationForecaster


def main():
    """Main execution function."""
    print("=" * 70)
    print("INFLATION FORECASTING WITH ELASTICNET")
    print("=" * 70)

    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data'
    api_key = os.environ.get('FRED_API_KEY', '')

    # Load data
    print("\nLoading data...")
    train_file = get_latest_vintage(data_dir)
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = \
        build_dataset_with_fred_target(train_file, api_key)

    print(f"Train samples : {len(X_train)}  features: {len(feature_names)}")
    print(f"Val samples   : {len(X_val)}")
    print(f"Test samples  : {len(X_test)}")

    # Train ElasticNet
    print("\nTraining ElasticNet...")
    forecaster = InflationForecaster()
    forecaster.train(X_train, y_train, tune_hyperparameters=True, cv_folds=5)

    # Evaluate
    val_metrics, _ = forecaster.evaluate(X_val, y_val)
    print("\nValidation performance:")
    print(f"  RMSE : {val_metrics['rmse']:.4f}")
    print(f"  MAE  : {val_metrics['mae']:.4f}")
    print(f"  R²   : {val_metrics['r2']:.4f}")
    print(f"  MAPE : {val_metrics['mape']:.2f}%")

    if len(X_test) > 0:
        test_cols = [c for c in feature_names if c in X_test.columns]
        test_metrics, _ = forecaster.evaluate(X_test[test_cols], y_test)
        print("\nTest performance:")
        print(f"  RMSE : {test_metrics['rmse']:.4f}")
        print(f"  MAE  : {test_metrics['mae']:.4f}")
        print(f"  R²   : {test_metrics['r2']:.4f}")
        print(f"  MAPE : {test_metrics['mape']:.2f}%")

    # Feature importance
    importance_df = forecaster.get_feature_importance(feature_names)
    if importance_df is not None:
        print("\nTop 10 features by |coefficient|:")
        print(importance_df.head(10).to_string(index=False))

    # Save results
    Path('results').mkdir(exist_ok=True)
    if importance_df is not None:
        importance_df.to_csv('results/feature_importance_elasticnet.csv', index=False)
        print("\nSaved: results/feature_importance_elasticnet.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
