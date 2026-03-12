"""
Macro-Econ-ML: Machine learning for macroeconomic forecasting.
"""

__version__ = '1.0.0'
__author__ = 'UVM Machine Learning Class'

from .data_loader import (
    load_fred_md_file, get_latest_vintage, build_dataset,
    fetch_fred_series, build_dataset_with_fred_target,
)
from .models import InflationForecaster

__all__ = [
    'load_fred_md_file',
    'get_latest_vintage',
    'build_dataset',
    'fetch_fred_series',
    'build_dataset_with_fred_target',
    'InflationForecaster',
]
