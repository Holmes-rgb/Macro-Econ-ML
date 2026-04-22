"""
fred_md_utils.py — Shared data utilities for FRED-MD inflation forecasting.

Provides data downloading, transformation, loading, and dataset construction
for any model notebook in this project. Import with:

    import sys
    sys.path.insert(0, '.')
    from fred_md_utils import (
        download_latest_vintage, get_latest_vintage,
        load_fred_md_file, build_dataset_from_csv, make_sequences,
    )
"""

import datetime
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
# --------------------------------------------------------------------------
# Constants for shared split dates — change here to propagate to every model
# --------------------------------------------------------------------------

TEST_START = '2024-01-01'
VAL_START  = '2010-01-01'

# ---------------------------------------------------------------------------
# Transformation helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Data loading functions
# ---------------------------------------------------------------------------

def load_fred_md_file(filepath):
    """
    Load a single FRED-MD CSV file.

    FRED-MD format:
      Row 1: transformation codes (tcode) for each series
      Row 2+: monthly observations

    Returns:
        Tuple (transformed_df, tcodes_dict)
    """
    filepath = str(filepath)
    raw_all = pd.read_csv(filepath, header=None)
    headers = raw_all.iloc[0].tolist()
    tcode_row = raw_all.iloc[1].tolist()

    tcodes = {}
    for h, t in zip(headers[1:], tcode_row[1:]):
        try:
            tcodes[str(h)] = float(t)
        except (ValueError, TypeError):
            pass

    df = raw_all.iloc[2:].copy()
    df.columns = headers
    df = df.reset_index(drop=True)

    date_col_actual = df.columns[0]
    df[date_col_actual] = pd.to_datetime(df[date_col_actual])
    df = df.set_index(date_col_actual)
    df.index.name = 'date'
    df = df.dropna(how='all')

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df_transformed = apply_fred_md_transforms(df, tcodes)
    return df_transformed, tcodes


def load_fred_md_raw(filepath):
    """Load FRED-MD CSV as raw (untransformed) price levels plus tcode dict."""
    filepath = str(filepath)
    raw_all = pd.read_csv(filepath, header=None)
    headers = raw_all.iloc[0].tolist()
    tcode_row = raw_all.iloc[1].tolist()
    tcodes = {}
    for h, t in zip(headers[1:], tcode_row[1:]):
        try:
            tcodes[str(h)] = float(t)
        except (ValueError, TypeError):
            pass
    df = raw_all.iloc[2:].copy()
    df.columns = headers
    df = df.reset_index(drop=True)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df.index.name = 'date'
    df = df.dropna(how='all')
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df, tcodes


def get_latest_vintage(vintage_dir):
    """
    Find the most recent vintage CSV file in a directory.

    Vintage files are named like '2025-01-MD.csv', '2024-12-MD.csv', etc.

    Returns:
        Path to the most recent vintage file
    """
    vintage_dir = Path(vintage_dir)
    csv_files = list(vintage_dir.glob('*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {vintage_dir}")
    csv_files_sorted = sorted(csv_files, key=lambda p: p.stem)
    latest = csv_files_sorted[-1]
    print(f"Using latest vintage: {latest.name}")
    return latest


def _is_valid_fred_md_csv(path):
    """Return True if the file looks like a real FRED-MD CSV (not an HTML error page)."""
    try:
        with open(path, 'r', errors='replace') as f:
            first_line = f.readline()
        lower = first_line.lower()
        return 'sasdate' in lower or (not lower.startswith('<') and ',' in first_line)
    except Exception:
        return False


def download_latest_vintage(data_dir):
    """Try current month then back up until a FRED-MD vintage CSV is found; download it.

    delta=0 (current month): always attempt download to ensure freshness and heal
    any corrupt/truncated cache. Falls back to cached copy on network failure or
    when the vintage hasn't been published yet.

    delta>=1 (older months): trust the cache — these are immutable snapshots and
    re-downloading is wasteful.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    base_url = ("https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed"
                "/research/fred-md/monthly/{year:04d}-{month:02d}-md.csv")

    today = datetime.date.today()
    for delta in range(6):
        candidate = today.replace(day=1) - relativedelta(months=delta)
        year, month = candidate.year, candidate.month
        filename = f"{year:04d}-{month:02d}-MD.csv"
        local_path = data_dir / filename

        if delta >= 1 and local_path.exists():
            if _is_valid_fred_md_csv(local_path):
                print(f"Already have {filename}")
                return local_path
            else:
                print(f"Removing invalid cached file {filename}")
                local_path.unlink()

        url = base_url.format(year=year, month=month)
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                snippet = resp.content[:200].decode('utf-8', errors='replace')
                if '<html' in snippet.lower() or '<!doctype' in snippet.lower():
                    print(f"  {filename}: URL returned HTML — vintage not yet published")
                    if delta == 0 and local_path.exists() and _is_valid_fred_md_csv(local_path):
                        print(f"  Using cached {filename}")
                        return local_path
                    continue
                local_path.write_bytes(resp.content)
                print(f"Downloaded {filename}")
                print(local_path)
                return local_path
        except requests.RequestException:
            if delta == 0 and local_path.exists() and _is_valid_fred_md_csv(local_path):
                print(f"  Network error — using cached {filename}")
                return local_path
    raise RuntimeError("Could not download any recent FRED-MD vintage.")


def build_dataset_from_csv(filepath, horizon=1, n_lags=2,
                            test_start=TEST_START, val_start=VAL_START):
    """
    Build train/val/test datasets from a single FRED-MD vintage CSV.

    Target: PCEPI first-log-diff x100 (monthly % change), shifted by `horizon`.
    Features: all columns transformed by their tcodes, plus n_lags lagged copies
    of every feature (e.g. n_lags=2 adds _lag1, _lag2 columns).
    Splits are date-based (no shuffling).

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, feature_names
    """
    filepath = str(filepath)
    raw_all = pd.read_csv(filepath, header=None)
    headers = raw_all.iloc[0].tolist()
    tcode_row = raw_all.iloc[1].tolist()

    tcodes = {}
    for h, t in zip(headers[1:], tcode_row[1:]):
        try:
            tcodes[str(h)] = float(t)
        except (ValueError, TypeError):
            pass

    df_raw = raw_all.iloc[2:].copy()
    df_raw.columns = headers
    df_raw = df_raw.reset_index(drop=True)
    date_col = df_raw.columns[0]
    df_raw[date_col] = pd.to_datetime(df_raw[date_col])
    df_raw = df_raw.set_index(date_col)
    df_raw.index.name = 'date'
    df_raw = df_raw.dropna(how='all')
    for col in df_raw.columns:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

    pcepi_raw = df_raw['PCEPI']
    target = np.log(pcepi_raw).diff() * 100
    target_shifted = target.shift(-horizon)

    df_features = apply_fred_md_transforms(df_raw, tcodes)
    df_features = df_features.ffill().bfill()

    if n_lags > 0:
        base_cols = df_features.columns.tolist()
        lag_frames = [df_features]
        for lag in range(1, n_lags + 1):
            lag_df = df_features[base_cols].shift(lag)
            lag_df.columns = [f"{c}_lag{lag}" for c in base_cols]
            lag_frames.append(lag_df)
        df_features = pd.concat(lag_frames, axis=1)

    df = df_features.copy()
    df['_target'] = target_shifted
    df = df.dropna()

    X = df.drop(columns=['_target'])
    y = df['_target']

    X = X.dropna(axis=1, how='all')
    feature_names = X.columns.tolist()

    test_mask  = X.index >= pd.Timestamp(test_start)
    val_mask   = (X.index >= pd.Timestamp(val_start)) & ~test_mask
    train_mask = ~val_mask & ~test_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    print(f"\nDataset summary (PCEPI first-log-diff target, {horizon}-month-ahead):")
    print(f"  Train : {len(X_train):4d} obs  {X_train.index.min().date()} -> {X_train.index.max().date()}")
    print(f"  Val   : {len(X_val):4d} obs  {X_val.index.min().date()} -> {X_val.index.max().date()}")
    print(f"  Test  : {len(X_test):4d} obs  {X_test.index.min().date()} -> {X_test.index.max().date()}")
    base_count = len([f for f in feature_names if '_lag' not in f])
    print(f"  Features: {len(feature_names)} ({base_count} base × {n_lags + 1} time steps)")

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names


# ---------------------------------------------------------------------------
# Sequence utilities (for RNN / LSTM models)
# ---------------------------------------------------------------------------

def make_sequences(X, y, seq_len):
    """
    Convert 2D feature array and target vector into overlapping 3D sequences
    for use with LSTM / RNN models.

    Each output sample uses the window X[i-seq_len+1 : i+1] to predict y[i],
    so the label is aligned to the *last* timestep of the window.

    Args:
        X       : np.ndarray of shape (n_samples, n_features) — already scaled
        y       : np.ndarray of shape (n_samples,)
        seq_len : int — number of consecutive timesteps per sequence

    Returns:
        X_seq : np.ndarray of shape (n_samples - seq_len + 1, seq_len, n_features)
        y_seq : np.ndarray of shape (n_samples - seq_len + 1,)

    Note:
        When building sequences for val/test splits, prepend the last (seq_len - 1)
        rows from the preceding split so that the first val/test sample still has a
        full context window:

            X_val_seq, y_val_seq = make_sequences(
                np.vstack([X_train_scaled[-(seq_len - 1):], X_val_scaled]),
                np.concatenate([y_train[-(seq_len - 1):], y_val]),
                seq_len,
            )
            # Then drop the first (seq_len - 1) samples which were only context:
            X_val_seq = X_val_seq[seq_len - 1:]
            y_val_seq = y_val_seq[seq_len - 1:]
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(X)
    if seq_len > n:
        raise ValueError(f"seq_len ({seq_len}) cannot exceed number of samples ({n})")
    X_seq = np.stack([X[i - seq_len + 1: i + 1] for i in range(seq_len - 1, n)])
    y_seq = y[seq_len - 1:]
    return X_seq, y_seq


# ---------------------------------------------------------------------------
# Notebook setup helpers
# ---------------------------------------------------------------------------

def default_paths(results_subdir='results'):
    """Return (VINTAGE_DIR, RESULTS_DIR), resolving for both notebooks/ and repo root.

    Creates RESULTS_DIR if missing.
    """
    import os
    vintage_dir = '../data' if os.path.exists('../data') else 'data'
    results_dir = results_subdir
    os.makedirs(results_dir, exist_ok=True)
    return vintage_dir, results_dir


def configure_plots(figsize=(13, 5), title_size=13, label_size=11):
    """Apply the shared matplotlib/seaborn style used across all notebooks."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use('default')
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['axes.titlesize'] = title_size
    plt.rcParams['axes.labelsize'] = label_size
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True


def get_splits(vintage_dir=None, horizon=1, n_lags=0,
               test_start=None, val_start=None):
    """Download/locate latest FRED-MD vintage and build train/val/test splits.

    Returns:
        (vintage_file, X_train, y_train, X_val, y_val, X_test, y_test, feature_names)
    """
    if vintage_dir is None:
        vintage_dir, _ = default_paths()
    if test_start is None:
        test_start = TEST_START
    if val_start is None:
        val_start = VAL_START
    vintage_file = download_latest_vintage(vintage_dir)
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = build_dataset_from_csv(
        filepath=vintage_file, horizon=horizon, n_lags=n_lags,
        test_start=test_start, val_start=val_start,
    )
    return vintage_file, X_train, y_train, X_val, y_val, X_test, y_test, feature_names
