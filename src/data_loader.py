"""
Data loader for FRED-MD inflation forecasting.
Loads user-provided FRED-MD CSV files, applies transformation codes,
and constructs train/test datasets for 1-month-ahead inflation forecasting.
"""

import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path


# FRED-MD transformation code definitions
# 1: no transform, 2: 1st diff, 3: 2nd diff, 4: log,
# 5: log 1st diff * 100, 6: log 2nd diff * 100, 7: delta(x/x_lag - 1)*100
def _apply_tcode(series, code):
    """Apply a single FRED-MD transformation code to a pandas Series."""
    code = int(code)
    s = series.copy().astype(float)
    if code == 1:
        return s
    elif code == 2:
        return s.diff()
    elif code == 3:
        return s.diff().diff()
    elif code == 4:
        return np.log(s)
    elif code == 5:
        return np.log(s).diff() * 100
    elif code == 6:
        return np.log(s).diff().diff() * 100
    elif code == 7:
        pct = s / s.shift(1) - 1
        return pct.diff() * 100
    else:
        warnings.warn(f"Unknown tcode {code}, returning series unchanged.")
        return s


def apply_fred_md_transforms(df, tcodes):
    """
    Apply FRED-MD transformation codes to each column.

    Args:
        df: DataFrame with raw series (dates as index)
        tcodes: dict mapping column name -> tcode (int or float)

    Returns:
        Transformed DataFrame (same shape, NaNs introduced at boundaries)
    """
    transformed = {}
    for col in df.columns:
        if col in tcodes:
            transformed[col] = _apply_tcode(df[col], tcodes[col])
        else:
            transformed[col] = df[col].copy()
    return pd.DataFrame(transformed, index=df.index)


def load_fred_md_file(filepath):
    """
    Load a single FRED-MD CSV file.

    FRED-MD format:
      Row 1: transformation codes (tcode) for each series
      Row 2+: monthly observations

    Args:
        filepath: path to the FRED-MD CSV

    Returns:
        Tuple (transformed_df, tcodes_dict)
        transformed_df: DataFrame indexed by date, with tcodes applied
        tcodes_dict: dict of {column: tcode}
    """
    filepath = str(filepath)
    # Read first row to get tcodes
    raw = pd.read_csv(filepath, header=None, nrows=2)
    # Row 0: column headers (first col is 'sasdate' or similar)
    # Row 1: tcode values
    headers = raw.iloc[0].tolist()
    tcode_row = raw.iloc[1].tolist()

    # Build tcode dict (skip the date column)
    date_col = headers[0]
    tcodes = {}
    for h, t in zip(headers[1:], tcode_row[1:]):
        try:
            tcodes[str(h)] = float(t)
        except (ValueError, TypeError):
            pass  # skip non-numeric tcode entries

    # Read actual data (skip tcode row, keep header row 0)
    df = pd.read_csv(filepath, skiprows=[1])

    # Parse date column
    date_col_actual = df.columns[0]
    df[date_col_actual] = pd.to_datetime(df[date_col_actual])
    df = df.set_index(date_col_actual)
    df.index.name = 'date'

    # Drop any trailing NaN-only rows (FRED-MD sometimes has a blank last row)
    df = df.dropna(how='all')

    # Convert all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Apply transformations
    df_transformed = apply_fred_md_transforms(df, tcodes)

    return df_transformed, tcodes


def get_latest_vintage(vintage_dir):
    """
    Find the most recent vintage CSV file in a directory.

    Vintage files are named like '2025-01.csv', '2024-12.csv', etc.
    Returns the path of the file with the latest date in its name.

    Args:
        vintage_dir: path to directory containing vintage CSV files

    Returns:
        Path to the most recent vintage file
    """
    vintage_dir = Path(vintage_dir)
    csv_files = list(vintage_dir.glob('*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {vintage_dir}")

    # Sort by filename (YYYY-MM format sorts lexicographically = chronologically)
    csv_files_sorted = sorted(csv_files, key=lambda p: p.stem)
    latest = csv_files_sorted[-1]
    print(f"Using latest vintage: {latest.name}")
    return latest


def build_dataset(train_filepath, test_filepath, target='PCEPI', horizon=1):
    """
    Build train and test datasets for h-step-ahead inflation forecasting.

    Features at time t predict target (PCEPI growth) at time t+horizon.

    Args:
        train_filepath: path to training FRED-MD CSV (e.g. latest vintage)
        test_filepath: path to test FRED-MD CSV (e.g. Feb 2026 current file)
        target: column name of the price index to forecast (default 'PCEPI')
        horizon: forecast horizon in months (default 1)

    Returns:
        X_train, y_train, X_test, y_test, feature_names
    """
    print(f"Loading training data from: {train_filepath}")
    df_train, tcodes_train = load_fred_md_file(train_filepath)

    print(f"Loading test data from: {test_filepath}")
    df_test, tcodes_test = load_fred_md_file(test_filepath)

    # Resolve target column: prefer PCEPI, fall back to CPIAUCSL
    def resolve_target(df, label):
        if target in df.columns:
            return target
        alt = 'CPIAUCSL'
        if alt in df.columns:
            warnings.warn(
                f"'{target}' not found in {label} file. Falling back to '{alt}'."
            )
            return alt
        raise KeyError(
            f"Neither '{target}' nor 'CPIAUCSL' found in {label} file. "
            f"Available columns: {df.columns.tolist()[:20]}"
        )

    train_target = resolve_target(df_train, 'train')
    test_target = resolve_target(df_test, 'test')

    # Build target series: transformed PCEPI growth, shifted by -horizon
    # so that y[t] = inflation at t+1 (1-month-ahead target)
    y_train_raw = df_train[train_target].shift(-horizon)
    y_test_raw = df_test[test_target].shift(-horizon)

    # Features: all columns except the target
    feature_cols_train = [c for c in df_train.columns if c != train_target]
    feature_cols_test = [c for c in df_test.columns if c != test_target]

    # Use intersection of feature columns for alignment
    common_features = [c for c in feature_cols_train if c in feature_cols_test]
    if not common_features:
        raise ValueError("No common feature columns between train and test files.")

    X_train_raw = df_train[common_features]
    X_test_raw = df_test[common_features]

    # Align X and y: drop rows where either is NaN
    # Also drop the last `horizon` rows of training (no target available due to shift)
    train_valid = (~y_train_raw.isna()) & (~X_train_raw.isna().all(axis=1))
    X_train = X_train_raw[train_valid].copy()
    y_train = y_train_raw[train_valid].copy()

    test_valid = (~y_test_raw.isna()) & (~X_test_raw.isna().all(axis=1))
    X_test = X_test_raw[test_valid].copy()
    y_test = y_test_raw[test_valid].copy()

    # Handle missing feature values: forward-fill then backward-fill within each split
    X_train = X_train.ffill().bfill()
    X_test = X_test.ffill().bfill()

    # Drop feature columns still all-NaN after fill
    X_train = X_train.dropna(axis=1, how='all')
    keep_cols = X_train.columns.tolist()
    X_test = X_test[keep_cols] if all(c in X_test.columns for c in keep_cols) else X_test

    # Restrict test set to observations after the training end date (no leakage)
    train_end = X_train.index.max()
    test_after = X_test.index > train_end
    if test_after.sum() == 0:
        warnings.warn(
            "Test set has no observations after the training end date. "
            "The test file may overlap entirely with the training vintage. "
            "Using all test observations anyway."
        )
    else:
        X_test = X_test[test_after]
        y_test = y_test[test_after]

    feature_names = X_train.columns.tolist()

    print(f"\nDataset summary:")
    print(f"  Training: {X_train.shape[0]} obs, {X_train.shape[1]} features")
    print(f"  Training date range: {X_train.index.min().date()} to {X_train.index.max().date()}")
    print(f"  Test:     {X_test.shape[0]} obs, {X_test.shape[1]} features")
    if len(X_test) > 0:
        print(f"  Test date range:     {X_test.index.min().date()} to {X_test.index.max().date()}")
    print(f"  Target: {train_target} (tcode={tcodes_train.get(train_target, 'N/A')}), {horizon}-month-ahead")

    return X_train, y_train, X_test, y_test, feature_names


def fetch_fred_series(series_id, api_key, start_date=None):
    """
    Fetch a time series from the FRED REST API.

    Args:
        series_id: FRED series identifier (e.g. 'PCEPI')
        api_key: FRED API key string
        start_date: optional start date string 'YYYY-MM-DD'

    Returns:
        pd.Series indexed by monthly period-end DatetimeIndex
    """
    import requests

    params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json',
    }
    if start_date is not None:
        params['observation_start'] = start_date

    url = 'https://api.stlouisfed.org/fred/series/observations'
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    observations = data.get('observations', [])
    if not observations:
        raise ValueError(f"No observations returned for series '{series_id}'")

    dates, values = [], []
    for obs in observations:
        if obs['value'] == '.':
            continue
        dates.append(pd.to_datetime(obs['date']))
        values.append(float(obs['value']))

    series = pd.Series(values, index=pd.DatetimeIndex(dates), name=series_id)
    # Snap to month-end so it aligns with FRED-MD dates
    series.index = series.index + pd.offsets.MonthEnd(0)
    return series


def build_dataset_with_fred_target(
    vintage_filepath,
    fred_api_key,
    target_series='PCEPI',
    horizon=1,
    train_frac=0.70,
    val_frac=0.15,
):
    """
    Build train / val / test datasets using a FRED-API-sourced target.

    Features come from a FRED-MD vintage CSV; the target is fetched live
    from the FRED REST API, log-differenced × 100, and shifted by `horizon`.

    Args:
        vintage_filepath: path to a FRED-MD vintage CSV
        fred_api_key: FRED API key string
        target_series: FRED series ID to use as target (default 'PCEPI')
        horizon: forecast horizon in months (default 1)
        train_frac: fraction of rows for training (default 0.70)
        val_frac: fraction of rows for validation (default 0.15)

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, feature_names
    """
    print(f"Loading features from: {vintage_filepath}")
    df_features, tcodes = load_fred_md_file(vintage_filepath)

    # FRED-MD dates are month-start; snap to month-end to match FRED API dates
    df_features.index = df_features.index + pd.offsets.MonthEnd(0)

    print(f"Fetching {target_series} from FRED API…")
    raw_target = fetch_fred_series(target_series, fred_api_key)

    # Apply log first-difference × 100 (tcode 6) to the raw target
    target_transformed = np.log(raw_target).diff() * 100
    target_transformed.name = target_series

    # Shift target by -horizon so y[t] = inflation at t+horizon
    target_shifted = target_transformed.shift(-horizon)

    # Merge features and target on date index
    df = df_features.copy()
    df[target_series + '_target'] = target_shifted.reindex(df.index)

    # Fill features: forward-fill then backward-fill to cover leading NaNs
    feature_cols = [c for c in df.columns if c != target_series + '_target']
    df[feature_cols] = df[feature_cols].ffill().bfill()

    # Drop rows where target is NaN
    df = df.dropna(subset=[target_series + '_target'])

    X = df[feature_cols]
    y = df[target_series + '_target']

    # Drop feature columns that are still all-NaN
    X = X.dropna(axis=1, how='all')
    feature_names = X.columns.tolist()

    # Temporal split (no shuffling)
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    X_train = X.iloc[:n_train]
    y_train = y.iloc[:n_train]
    X_val = X.iloc[n_train:n_train + n_val]
    y_val = y.iloc[n_train:n_train + n_val]
    X_test = X.iloc[n_train + n_val:]
    y_test = y.iloc[n_train + n_val:]

    print(f"\nDataset summary ({target_series} target, {horizon}-month-ahead):")
    print(f"  Train : {len(X_train):4d} obs  {X_train.index.min().date()} → {X_train.index.max().date()}")
    print(f"  Val   : {len(X_val):4d} obs  {X_val.index.min().date()} → {X_val.index.max().date()}")
    print(f"  Test  : {len(X_test):4d} obs  {X_test.index.min().date()} → {X_test.index.max().date()}")
    print(f"  Features: {len(feature_names)}")

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names
