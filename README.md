# Macro-Econ-ML

Machine learning models on macroeconomic data for inflation forecasting.

## Overview

This project uses scikit-learn to evaluate various machine learning models for forecasting inflation in the US economy. The primary focus is on **ElasticNet regression**, with additional comparisons to Ridge, Lasso, Random Forest, and Gradient Boosting models.

## Data Sources

- **Training Data**: FRED-MD and FRED-QD databases from the Federal Reserve Bank of St. Louis
  - URL: https://www.stlouisfed.org/research/economists/mccracken/fred-databases
  - These datasets contain monthly and quarterly macroeconomic indicators

- **Real-time Inflation Data**: Personal Consumption Expenditures (PCE) data
  - URL: https://fred.stlouisfed.org/series/DPCCRV1Q225SBEA

## Features

- **Automated Data Download**: Fetches latest FRED economic data
- **Multiple Models**: Compares ElasticNet, Ridge, Lasso, Random Forest, and Gradient Boosting
- **Hyperparameter Tuning**: Grid search with time series cross-validation
- **Comprehensive Evaluation**: RMSE, MAE, R², and MAPE metrics
- **Visualizations**: Prediction plots, model comparisons, residual analysis, and feature importance
- **Time Series Aware**: Uses appropriate train/test splits for temporal data

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Holmes-rgb/Macro-Econ-ML.git
cd Macro-Econ-ML
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the complete pipeline with a single command:

```bash
python main.py
```

This will:
1. Download the latest FRED economic data (or use sample data if offline)
2. Preprocess the data
3. Train and evaluate all models (ElasticNet, Ridge, Lasso, Random Forest, Gradient Boosting)
4. Generate visualizations
5. Save results to the `results/` directory

### Interactive Analysis with Jupyter

For interactive exploration and analysis:

```bash
jupyter notebook notebooks/inflation_forecasting.ipynb
```

The notebook includes:
- Step-by-step data loading and preprocessing
- Exploratory data analysis with visualizations
- ElasticNet model training and evaluation
- Model comparison across all algorithms
- Feature importance analysis

### Using Individual Modules

#### Data Loading
```python
from src.data_loader import download_fred_data, load_and_preprocess_data

# Download data
download_fred_data(save_path='data/')

# Load and preprocess
X, y, features = load_and_preprocess_data(data_path='data/fred_qd.csv')
```

#### Training a Single Model
```python
from src.models import InflationForecaster

# Initialize ElasticNet model
forecaster = InflationForecaster(model_type='elasticnet')

# Train with hyperparameter tuning
forecaster.train(X_train, y_train, tune_hyperparameters=True)

# Make predictions
predictions = forecaster.predict(X_test)

# Evaluate
metrics, y_pred = forecaster.evaluate(X_test, y_test)
```

#### Comparing Multiple Models
```python
from src.models import compare_models

results_df, predictions, X_test, y_test = compare_models(X, y, test_size=0.2)
print(results_df)
```

#### Visualizations
```python
from src.visualizations import plot_predictions, plot_model_comparison

# Plot predictions
plot_predictions(y_test, predictions, dates=y_test.index, save_path='predictions.png')

# Plot model comparison
plot_model_comparison(results_df, save_path='comparison.png')
```

## Project Structure

```
Macro-Econ-ML/
├── data/                      # Downloaded data files
│   ├── fred_md.csv           # Monthly data
│   └── fred_qd.csv           # Quarterly data
├── src/                       # Source code
│   ├── data_loader.py        # Data download and preprocessing
│   ├── models.py             # Model training and evaluation
│   └── visualizations.py     # Plotting functions
├── results/                   # Output files
│   ├── model_comparison.csv
│   ├── predictions_comparison.png
│   ├── model_comparison.png
│   └── feature_importance_*.png
├── notebooks/                 # Jupyter notebooks (if any)
│   └── inflation_forecasting.ipynb  # Interactive analysis notebook
├── main.py                    # Main entry point
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Models

### ElasticNet (Primary Model)
- Combines L1 (Lasso) and L2 (Ridge) regularization
- Hyperparameters: `alpha` (regularization strength), `l1_ratio` (balance between L1 and L2)
- Good for high-dimensional data with correlated features

### Other Models
- **Ridge**: L2 regularization, handles multicollinearity
- **Lasso**: L1 regularization, performs feature selection
- **Random Forest**: Ensemble of decision trees, captures non-linear relationships
- **Gradient Boosting**: Sequential ensemble, often achieves highest accuracy

## Evaluation Metrics

- **RMSE** (Root Mean Squared Error): Lower is better
- **MAE** (Mean Absolute Error): Lower is better  
- **R²** (R-squared): Higher is better (closer to 1)
- **MAPE** (Mean Absolute Percentage Error): Lower is better

## Results

The results will be saved to the `results/` directory including:
- Model comparison CSV file
- Prediction plots
- Model performance comparison charts
- Residual analysis plots
- Feature importance rankings

## Requirements

- Python 3.8+
- scikit-learn >= 1.3.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- requests >= 2.31.0

## Note on Data Availability

If you're unable to download data from FRED (e.g., in a restricted network environment), you can generate synthetic sample data:

```bash
python src/generate_sample_data.py
```

This will create realistic economic time series data in the `data/` directory for testing purposes.

## Contributing

This is a class project for machine learning at UVM. Feel free to fork and experiment with different models or feature engineering approaches.

## License

MIT License

## Contact

For questions or issues, please open an issue on GitHub.
