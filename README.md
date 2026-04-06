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
- **ElasticNet Model**: L1+L2 regularized regression optimized for macroeconomic forecasting
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

Open the notebook for the full interactive pipeline:

```bash
jupyter notebook notebooks/inflation_forecasting_elastinet.ipynb
```

The notebook includes:
- Data loading and FRED-MD transformation helpers (self-contained, no external src/ module)
- Exploratory data analysis with visualizations
- ElasticNet model training with TimeSeriesSplit CV
- Out-of-sample evaluation and feature importance analysis

## Project Structure

```
Macro-Econ-ML/
├── data/                      # FRED-MD vintage CSV files
├── notebooks/
│   └── inflation_forecasting.ipynb  # Full pipeline (self-contained)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Models

### ElasticNet (Primary Model)
- Combines L1 (Lasso) and L2 (Ridge) regularization
- Hyperparameters: `alpha` (regularization strength), `l1_ratio` (balance between L1 and L2)
- Good for high-dimensional data with correlated features

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
