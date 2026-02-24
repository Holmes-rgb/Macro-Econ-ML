"""
Macro-Econ-ML: Machine learning for macroeconomic forecasting.
"""

__version__ = '1.0.0'
__author__ = 'UVM Machine Learning Class'

from .data_loader import download_fred_data, load_and_preprocess_data
from .models import InflationForecaster, compare_models
from .visualizations import (
    plot_predictions, 
    plot_model_comparison, 
    plot_residuals, 
    plot_feature_importance
)

__all__ = [
    'download_fred_data',
    'load_and_preprocess_data',
    'InflationForecaster',
    'compare_models',
    'plot_predictions',
    'plot_model_comparison',
    'plot_residuals',
    'plot_feature_importance',
]
