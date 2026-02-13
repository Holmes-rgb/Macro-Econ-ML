"""
Data loader for FRED economic data.
Downloads and processes training data from FRED database.
"""

import pandas as pd
import requests
import os
from pathlib import Path


def download_fred_data(save_path='data/'):
    """
    Download FRED-MD (Monthly) and FRED-QD (Quarterly) datasets.
    These datasets are commonly used for macroeconomic forecasting.
    
    FRED-MD: Monthly data
    FRED-QD: Quarterly data
    
    Args:
        save_path: Directory to save downloaded data
    """
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # URLs for FRED databases (CSV format)
    # These are the McCracken FRED databases mentioned in the problem statement
    fred_md_url = "https://files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv"
    fred_qd_url = "https://files.stlouisfed.org/files/htdocs/fred-md/quarterly/current.csv"
    
    print("Downloading FRED-MD (Monthly data)...")
    try:
        response = requests.get(fred_md_url, timeout=30)
        response.raise_for_status()
        with open(os.path.join(save_path, 'fred_md.csv'), 'wb') as f:
            f.write(response.content)
        print(f"✓ FRED-MD downloaded to {save_path}fred_md.csv")
    except Exception as e:
        print(f"✗ Failed to download FRED-MD: {e}")
    
    print("\nDownloading FRED-QD (Quarterly data)...")
    try:
        response = requests.get(fred_qd_url, timeout=30)
        response.raise_for_status()
        with open(os.path.join(save_path, 'fred_qd.csv'), 'wb') as f:
            f.write(response.content)
        print(f"✓ FRED-QD downloaded to {save_path}fred_qd.csv")
    except Exception as e:
        print(f"✗ Failed to download FRED-QD: {e}")


def load_and_preprocess_data(data_path='data/fred_qd.csv', target_variable='DPCCRG3Q086SBEA'):
    """
    Load and preprocess FRED data for inflation forecasting.
    
    Args:
        data_path: Path to the FRED data file
        target_variable: The variable to forecast (default is real GDP growth rate)
                        For inflation, common variables are:
                        - CPIAUCSL: Consumer Price Index
                        - PCEPI: Personal Consumption Expenditures Price Index
                        - DPCCRG3Q086SBEA: Real GDP growth
    
    Returns:
        X: Feature matrix
        y: Target variable
        feature_names: List of feature names
    """
    print(f"\nLoading data from {data_path}...")
    
    # Read the FRED data
    # First row is often transformation codes, second row is headers
    df = pd.read_csv(data_path)
    
    # Skip the first row (transformation codes) if present
    if df.iloc[0].str.contains('Transform').any() or df.columns[0] == 'sasdate':
        df = pd.read_csv(data_path, skiprows=1)
    
    # Set date as index
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    
    # Convert all columns to numeric, coercing errors
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Check if target variable exists
    if target_variable not in df.columns:
        print(f"\nWarning: Target variable '{target_variable}' not found in dataset.")
        print("Available variables (first 20):")
        print(df.columns[:20].tolist())
        # Use first available numeric column as target
        target_variable = df.columns[0]
        print(f"Using '{target_variable}' as target variable.")
    
    # Separate features and target
    y = df[target_variable].copy()
    X = df.drop(columns=[target_variable])
    
    # Remove rows with missing target values
    valid_idx = ~y.isna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    # Handle missing values in features
    # Forward fill then backward fill
    X = X.fillna(method='ffill').fillna(method='bfill')
    
    # If still any NaN values, drop those features
    X = X.dropna(axis=1)
    
    print(f"\nProcessed data shape: X={X.shape}, y={y.shape}")
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    
    return X, y, X.columns.tolist()


if __name__ == "__main__":
    # Download data
    download_fred_data()
    
    # Load and preprocess
    try:
        X, y, features = load_and_preprocess_data()
        print(f"\nSuccessfully loaded data with {len(features)} features.")
    except Exception as e:
        print(f"\nError loading data: {e}")
