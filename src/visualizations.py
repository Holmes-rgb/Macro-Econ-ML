"""
Visualization utilities for model evaluation and results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path


def setup_plot_style():
    """Set up matplotlib style for consistent plots."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10


def plot_predictions(y_true, predictions_dict, dates=None, save_path=None):
    """
    Plot actual vs predicted values for multiple models.
    
    Args:
        y_true: Actual target values
        predictions_dict: Dictionary of {model_name: predictions}
        dates: Optional dates for x-axis
        save_path: Path to save the plot
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot actual values
    if dates is not None:
        ax.plot(dates, y_true, 'o-', label='Actual', linewidth=2, markersize=6, color='black')
    else:
        ax.plot(y_true, 'o-', label='Actual', linewidth=2, markersize=6, color='black')
    
    # Plot predictions from each model
    colors = plt.cm.tab10(np.linspace(0, 1, len(predictions_dict)))
    for (model_name, predictions), color in zip(predictions_dict.items(), colors):
        if dates is not None:
            ax.plot(dates, predictions, '--', label=f'{model_name.title()}', 
                   linewidth=2, alpha=0.7, color=color)
        else:
            ax.plot(predictions, '--', label=f'{model_name.title()}', 
                   linewidth=2, alpha=0.7, color=color)
    
    ax.set_xlabel('Time' if dates is not None else 'Sample Index', fontsize=12)
    ax.set_ylabel('Target Value', fontsize=12)
    ax.set_title('Actual vs Predicted Values', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_model_comparison(results_df, save_path=None):
    """
    Create bar plots comparing different models' performance.
    
    Args:
        results_df: DataFrame with model results
        save_path: Path to save the plot
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['rmse', 'mae', 'r2', 'mape']
    titles = ['Root Mean Squared Error', 'Mean Absolute Error', 'R² Score', 'Mean Absolute Percentage Error']
    
    for ax, metric, title in zip(axes.flat, metrics, titles):
        sorted_df = results_df.sort_values(metric, ascending=(metric != 'r2'))
        
        colors = ['green' if i == 0 else 'steelblue' for i in range(len(sorted_df))]
        
        ax.barh(sorted_df['model'], sorted_df[metric], color=colors, alpha=0.7)
        ax.set_xlabel(metric.upper(), fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, v in enumerate(sorted_df[metric]):
            ax.text(v, i, f' {v:.4f}', va='center', fontsize=9)
    
    plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_residuals(y_true, y_pred, model_name='Model', save_path=None):
    """
    Plot residuals analysis.
    
    Args:
        y_true: Actual target values
        y_pred: Predicted values
        model_name: Name of the model
        save_path: Path to save the plot
    """
    setup_plot_style()
    
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Residuals plot
    axes[0].scatter(y_pred, residuals, alpha=0.6)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Values', fontsize=11)
    axes[0].set_ylabel('Residuals', fontsize=11)
    axes[0].set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Histogram of residuals
    axes[1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Residuals', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Residuals Distribution', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f'Residual Analysis - {model_name}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_feature_importance(importance_df, top_n=20, save_path=None):
    """
    Plot feature importance.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to display
        save_path: Path to save the plot
    """
    setup_plot_style()
    
    top_features = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.barh(range(len(top_features)), top_features['importance'], color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_time_series_split(X, y, n_splits=5, save_path=None):
    """
    Visualize time series cross-validation splits.
    
    Args:
        X: Feature matrix (with DateTimeIndex)
        y: Target variable
        n_splits: Number of CV splits
        save_path: Path to save the plot
    """
    from sklearn.model_selection import TimeSeriesSplit
    
    setup_plot_style()
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        # Plot train split
        ax.barh(i, len(train_idx), left=0, height=0.5, color='blue', alpha=0.6, label='Train' if i == 0 else '')
        # Plot test split
        ax.barh(i, len(test_idx), left=len(train_idx), height=0.5, color='red', alpha=0.6, label='Test' if i == 0 else '')
    
    ax.set_yticks(range(n_splits))
    ax.set_yticklabels([f'Split {i+1}' for i in range(n_splits)])
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_title('Time Series Cross-Validation Splits', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Visualization module loaded successfully.")
    print("Available functions:")
    print("  - plot_predictions()")
    print("  - plot_model_comparison()")
    print("  - plot_residuals()")
    print("  - plot_feature_importance()")
    print("  - plot_time_series_split()")
