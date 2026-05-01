# Macro-Econ-ML

## Overview

This project evaluates multiple machine learning models for forecasting US inflation using the FRED-MD dataset. Models include ElasticNet , AR-OLS, LSTM, and a soft-voting ensemble. A dedicated comparison notebook evaluates all models on an identical out-of-sample test split.

## Data Sources

- **FRED-MD** — Monthly macroeconomic database from the Federal Reserve Bank of St. Louis
  - URL: https://www.stlouisfed.org/research/economists/mccracken/fred-databases
  - Vintage CSVs are auto-fetched if the current month's file is missing from `notebooks/data/`
- **Target series**: `PCEPI` (Personal Consumption Expenditures Price Index), transformed to monthly log-differences per FRED-MD tcode

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Holmes-rgb/Macro-Econ-ML.git
cd Macro-Econ-ML
```

2. Install dependencies (Python 3.14+ required):

**Recommended — using `uv`: (if you haven't tried uv yet, I highly recommend)**

```bash
uv sync
```

**Alternative — using pip:**

```bash
pip install -r requirements.txt
```

## Usage

Launch Jupyter and open any notebook:

```bash
# with uv
uv run jupyter notebook

# or with pip
jupyter notebook
```
Each notebook has a specific focus, and only runs one model. In order to run the model comparison notebook, you must first run all of the model notebooks. This can either be done manually, or by uncommenting and running the first cell in the `model_comparison.ipynb` notebook, which will automatically run all the other notebooks.

| Notebook | Purpose |
|---|---|
| `inflation_forecasting_elastinet.ipynb` | ElasticNet model — primary pipeline with data loading, feature engineering, tuning, and evaluation |
| `inflation_forecasting_ols.ipynb` | OLS baseline and random walk |
| `inflation_forecasting_rnn.ipynb` | Recurrent neural network model (LSTM) |
| `ensemble_soft_voting.ipynb` | Soft-voting ensemble of multiple models |
| `model_comparison.ipynb` | evaluates all models on the same test split |

## Project Structure

```
Macro-Econ-ML/
├── notebooks/
│   ├── data/                                    # FRED-MD vintage CSVs (auto-fetched)
│   │   ├── 2026-01-MD.csv
│   ├── results/                                 # Saved plots and comparison outputs
│   ├── fred_md_utils.py                         # Shared data-loading and helpers
│   ├── inflation_forecasting_elastinet.ipynb
│   ├── inflation_forecasting_ols.ipynb
│   ├── inflation_forecasting_rnn.ipynb
│   ├── ensemble_soft_voting.ipynb
│   └── model_comparison.ipynb
├── requirements.txt
└── README.md
```

## Evaluation Metrics

All metrics are computed on the same held-out test split and reported on the percentage points of monthly PCE inflation. An RMSE of 0.10 means forecasts are off by 0.10 percentage points per month on average.

| Metric | Better when |
|---|---|
| **RMSE** (Root Mean Squared Error) | Lower |
| **MAE** (Mean Absolute Error) | Lower |
| **R²** (R-squared) | Higher (max 1.0) |
| **MAPE** (Mean Absolute Percentage Error) | Lower |
