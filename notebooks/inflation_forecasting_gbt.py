# Used for testing effect of different params on the model, finding ranges to use in ensemble_voting
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.ensemble import (
    HistGradientBoostingRegressor
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

# base values to be used between tests 
# max_iter = 1000
# learning_rate = 0.05
# max_leaf nodes = 31
# min_samples_leaf = 20

# Finding best learning rate 

n_lr = 25
learning_rates = [(i, lr) for i, lr in enumerate(np.random.uniform(0.01,0.2, n_lr))]
lr_results = []

for i, lr in learning_rates:
    model = HistGradientBoostingRegressor(
        max_iter=1000,
        learning_rate=lr,
        max_leaf_nodes=31,        
        min_samples_leaf=20,      
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25,
        random_state=42
    )

    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    val_rmse  = np.sqrt(mean_squared_error(y_val,  val_pred))
    lr_results.append({'id': i, 'learning_rate': lr, 'val_rmse': val_rmse, 'test_rmse': test_rmse})
    print(f"ID={i}  lr={lr}  Val RMSE={val_rmse:.4f}  Test RMSE={test_rmse:.4f}")

best_lr = min(lr_results, key=lambda x: x['val_rmse'])
print(f"Best learning rate: {best_lr['learning_rate']}  Val RMSE={best_lr['val_rmse']:.4f}")

# Finding best max_iter
# Found that max_iter is not worth changing the value of, very little change in 
"""
n_iters = 10
max_iters = [(i, iters) for i, iters in enumerate(np.random.randint(100,2000, n_iters))]
iters_results = []

for i, iters in max_iters:
    model = HistGradientBoostingRegressor(
        max_iter=iters,
        learning_rate=.05,
        max_leaf_nodes=31,        
        min_samples_leaf=20,      
        l2_regularization=0.1,
        early_stopping=False,
        validation_fraction=0.15,
        n_iter_no_change=25,
        random_state=42
    )

    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    val_rmse  = np.sqrt(mean_squared_error(y_val,  val_pred))
    iters_results.append({'id': i, 'max_iter': iters, 'val_rmse': val_rmse, 'test_rmse': test_rmse})
    print(f"ID={i}  iters={iters}  Val RMSE={val_rmse:.4f}  Test RMSE={test_rmse:.4f}")

best_iters = min(iters_results, key=lambda x: x['val_rmse'])
print(f"Best learning rate: {best_iters['max_iter']}  Val RMSE={best_iters['val_rmse']:.4f}")"""

# Finding best learning rate 
# Doesn't change per run but used to see which values are the best to use in the ensemble
max_depth = [(i, d) for i, d in enumerate(range(3, 11))]
md_results = []

for i, d in max_depth:
    model = HistGradientBoostingRegressor(
        max_iter=1000,
        learning_rate=.05,
        max_depth=d,
        max_leaf_nodes=31,        
        min_samples_leaf=20,      
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25,
        random_state=42
    )

    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    val_rmse  = np.sqrt(mean_squared_error(y_val,  val_pred))
    md_results.append({'id': i, 'max_depth': d, 'val_rmse': val_rmse, 'test_rmse': test_rmse})
    print(f"ID={i}  d={d}  Val RMSE={val_rmse:.4f}  Test RMSE={test_rmse:.4f}")

best_md = min(md_results, key=lambda x: x['val_rmse'])
print(f"Best max depth: {best_md['max_depth']}  Val RMSE={best_md['val_rmse']:.4f}")

model = HistGradientBoostingRegressor(
        max_iter = 1000,
        learning_rate = best_lr['learning_rate'],
        max_depth=best_md['max_depth'],
        max_leaf_nodes=31,        
        min_samples_leaf=20,      
        l2_regularization=0.1,
        early_stopping=False,
        validation_fraction=0.15,
        n_iter_no_change=25,
        random_state=42
)
    
model.fit(X_train, y_train)

val_pred = model.predict(X_val)
test_pred = model.predict(X_test)

test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
val_rmse  = np.sqrt(mean_squared_error(y_val,  val_pred))
naive_rmse = np.sqrt(mean_squared_error(y_val.iloc[1:], y_val.shift(1).dropna()))

print(f"Val  RMSE: {val_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Native RMSE: {naive_rmse:.4f}")

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# --- Validation set ---
axes[0].plot(y_val.index, y_val.values, label='Actual', color='black')
axes[0].plot(y_val.index, val_pred,     label='Predicted', color='steelblue', linestyle='--')
axes[0].set_title('Validation Set: PCEPI Monthly % Change (1-month ahead)')
axes[0].set_ylabel('% Change')
axes[0].legend()
axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
axes[0].tick_params(axis='x', rotation=45)

# --- Test set ---
axes[1].plot(y_test.index, y_test.values, label='Actual', color='black')
axes[1].plot(y_test.index, test_pred,     label='Predicted', color='tomato', linestyle='--')
axes[1].set_title('Test Set: PCEPI Monthly % Change (1-month ahead)')
axes[1].set_ylabel('% Change')
axes[1].legend()
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

X_last = X_test.iloc[[-1]]  # most recent row
future_pred = model.predict(X_last)
print(f"1-month ahead PCEPI forecast: {future_pred[0]:.4f}%")