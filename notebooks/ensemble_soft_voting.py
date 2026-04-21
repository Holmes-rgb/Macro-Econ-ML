import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from itertools import product
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    GradientBoostingClassifier, 
    GradientBoostingRegressor
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from fred_md_utils import (
    download_latest_vintage,
    get_latest_vintage,
    build_dataset_from_csv,
    make_sequences,
)

TEST_START = '2025-06-01'
VAL_START  = '2023-01-01'

VINTAGE_DIR = '../data' if os.path.exists('../data') else 'data'

vintage_file = download_latest_vintage(VINTAGE_DIR)


X_train, y_train, X_val, y_val, X_test, y_test, feature_names = build_dataset_from_csv(
    filepath=vintage_file,
    horizon=1,
    n_lags=0,
    test_start=TEST_START,
    val_start=VAL_START,
)

max_iter = [100, 500, 1000, 2500]
learning_rate = [.01, .05, .1, .2]
max_leaf_nodes = [10, 20, 30, 40, 50]
min_samples_leaf = [10, 20, 50]

preds_sum = np.zeros(len(y_val))
preds_sum_test = np.zeros(len(y_test))
n_models  = 0

for max_iter, learning_rate, max_leaf_nodes, min_samples_leaf in product(max_iter, learning_rate, max_leaf_nodes, min_samples_leaf):
    model = HistGradientBoostingRegressor(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,        
        min_samples_leaf=min_samples_leaf,      
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25,
        random_state=42
    )
    model.fit(X_train, y_train)
    preds_sum += model.predict(X_val)
    preds_sum_test += model.predict(X_test)
    n_models  += 1

    if n_models % 10 == 0:
        print(f"  {n_models} models done...")

ensemble_pred = preds_sum / n_models
ensemble_test_pred = preds_sum_test / n_models
test_rmse = np.sqrt(mean_squared_error(y_test, ensemble_test_pred))
val_rmse  = np.sqrt(mean_squared_error(y_val,  ensemble_pred))
naive_rmse = np.sqrt(mean_squared_error(y_val.iloc[1:], y_val.shift(1).dropna()))



print(f"Ran {n_models} models")
print(f"Ensemble MAE: {mean_absolute_error(y_val, ensemble_pred):.4f}")
print(f"Naive RMSE: {naive_rmse:.4f}")
print(f"Val  RMSE: {val_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
