"""
Main script for inflation forecasting using machine learning.

This script downloads FRED economic data and trains multiple models
to forecast inflation, with a focus on ElasticNet as requested.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import download_fred_data, load_and_preprocess_data
from models import compare_models, InflationForecaster
from visualizations import (plot_predictions, plot_model_comparison, 
                           plot_residuals, plot_feature_importance)
import pandas as pd


def main():
    """Main execution function."""
    print("="*70)
    print("INFLATION FORECASTING WITH MACHINE LEARNING")
    print("="*70)
    
    # Step 1: Download data
    print("\n" + "="*70)
    print("STEP 1: Downloading FRED Economic Data")
    print("="*70)
    download_fred_data(save_path='data/')
    
    # Step 2: Load and preprocess data
    print("\n" + "="*70)
    print("STEP 2: Loading and Preprocessing Data")
    print("="*70)
    
    # Try to load quarterly data first (better for inflation forecasting)
    try:
        X, y, features = load_and_preprocess_data(
            data_path='data/fred_qd.csv'
        )
        data_frequency = "Quarterly"
    except Exception as e:
        print(f"Could not load quarterly data: {e}")
        print("Trying monthly data...")
        try:
            X, y, features = load_and_preprocess_data(
                data_path='data/fred_md.csv'
            )
            data_frequency = "Monthly"
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Please check that the data was downloaded correctly.")
            return
    
    print(f"\nUsing {data_frequency} data")
    print(f"Total features: {len(features)}")
    print(f"Total samples: {len(X)}")
    
    # Step 3: Train and compare models
    print("\n" + "="*70)
    print("STEP 3: Training and Comparing Models")
    print("="*70)
    print("Models to evaluate: ElasticNet (primary), Ridge, Lasso, Random Forest, Gradient Boosting")
    
    results_df, predictions, X_test, y_test = compare_models(
        X, y, test_size=0.2, random_state=42
    )
    
    # Step 4: Display results
    print("\n" + "="*70)
    print("STEP 4: Results Summary")
    print("="*70)
    print("\nModel Performance Comparison:")
    print(results_df.to_string(index=False))
    
    best_model = results_df.iloc[0]['model']
    print(f"\n⭐ Best performing model: {best_model.upper()}")
    print(f"   RMSE: {results_df.iloc[0]['rmse']:.4f}")
    print(f"   MAE:  {results_df.iloc[0]['mae']:.4f}")
    print(f"   R²:   {results_df.iloc[0]['r2']:.4f}")
    print(f"   MAPE: {results_df.iloc[0]['mape']:.2f}%")
    
    # Save results
    Path('results').mkdir(exist_ok=True)
    results_df.to_csv('results/model_comparison.csv', index=False)
    print("\n✓ Results saved to results/model_comparison.csv")
    
    # Step 5: Generate visualizations
    print("\n" + "="*70)
    print("STEP 5: Generating Visualizations")
    print("="*70)
    
    try:
        # Plot 1: Predictions comparison
        print("Creating predictions plot...")
        plot_predictions(
            y_test.values, 
            predictions, 
            dates=y_test.index,
            save_path='results/predictions_comparison.png'
        )
        
        # Plot 2: Model comparison
        print("Creating model comparison plot...")
        plot_model_comparison(
            results_df,
            save_path='results/model_comparison.png'
        )
        
        # Plot 3: Residuals for best model
        print(f"Creating residuals plot for {best_model}...")
        plot_residuals(
            y_test.values,
            predictions[best_model],
            model_name=best_model.title(),
            save_path=f'results/residuals_{best_model}.png'
        )
        
        # Plot 4: Feature importance for best model
        print(f"Creating feature importance plot for {best_model}...")
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        
        forecaster = InflationForecaster(model_type=best_model, random_state=42)
        forecaster.train(X_train, y_train, tune_hyperparameters=False)
        
        importance_df = forecaster.get_feature_importance(features)
        if importance_df is not None:
            plot_feature_importance(
                importance_df,
                top_n=20,
                save_path=f'results/feature_importance_{best_model}.png'
            )
            importance_df.head(20).to_csv(f'results/feature_importance_{best_model}.csv', index=False)
        
        print("\n✓ All visualizations saved to results/ directory")
        
    except Exception as e:
        print(f"Note: Could not generate all visualizations: {e}")
        print("This may be due to matplotlib backend issues in headless environment.")
    
    # Step 6: Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✓ Downloaded {data_frequency.lower()} economic data from FRED")
    print(f"✓ Preprocessed {len(features)} economic indicators")
    print(f"✓ Trained and evaluated 5 different models")
    print(f"✓ Best model: {best_model.upper()} with RMSE of {results_df.iloc[0]['rmse']:.4f}")
    print(f"✓ Results saved to results/ directory")
    
    print("\n" + "="*70)
    print("ElasticNet Performance (Primary Model):")
    print("="*70)
    elasticnet_results = results_df[results_df['model'] == 'elasticnet']
    if not elasticnet_results.empty:
        print(f"  RMSE: {elasticnet_results.iloc[0]['rmse']:.4f}")
        print(f"  MAE:  {elasticnet_results.iloc[0]['mae']:.4f}")
        print(f"  R²:   {elasticnet_results.iloc[0]['r2']:.4f}")
        print(f"  MAPE: {elasticnet_results.iloc[0]['mape']:.2f}%")
        print(f"  Best parameters: {elasticnet_results.iloc[0]['best_params']}")
    
    print("\n✨ Analysis complete!")


if __name__ == "__main__":
    main()
