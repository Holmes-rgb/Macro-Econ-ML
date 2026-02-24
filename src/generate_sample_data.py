"""
Create sample FRED-like data for testing when real data is unavailable.
This generates synthetic economic data with similar structure to FRED datasets.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_sample_data(n_periods=100, n_features=50, frequency='Q', save_path='data/'):
    """
    Generate synthetic economic time series data similar to FRED format.
    
    Args:
        n_periods: Number of time periods
        n_features: Number of economic indicators
        frequency: 'Q' for quarterly, 'M' for monthly
        save_path: Directory to save the data
    """
    np.random.seed(42)
    
    # Create date range
    if frequency == 'Q':
        start_date = datetime(1960, 1, 1)
        dates = pd.date_range(start=start_date, periods=n_periods, freq='QE')
        filename = 'fred_qd.csv'
    else:
        start_date = datetime(1960, 1, 1)
        dates = pd.date_range(start=start_date, periods=n_periods, freq='ME')
        filename = 'fred_md.csv'
    
    # Generate feature names (realistic economic indicators)
    feature_names = [
        'GDP', 'UNRATE', 'CPIAUCSL', 'PCEPI', 'INDPRO', 'PAYEMS',
        'HOUST', 'RSXFS', 'DSPIC96', 'PCE', 'M2SL', 'TB3MS',
        'GS10', 'FEDFUNDS', 'EXUSEU', 'EXUSUK', 'EXJPUS', 'UMCSENT',
        'VIXCLS', 'DCOILWTICO', 'DEXUSEU', 'DEXCHUS', 'DEXJPUS', 'AAA',
        'BAA', 'COMPAPFF', 'TB6MS', 'GS1', 'GS5', 'GS20',
        'MORTGAGE30US', 'DPRIME', 'BOGMBASE', 'TOTRESNS', 'BUSLOANS',
        'CONSUMER', 'REALLN', 'TOTALSL', 'LOANS', 'INVEST',
        'CP3M', 'WPSFD49207', 'WPSID61', 'PPIIDC', 'CPIAPPSL',
        'CPIULFSL', 'CPITRNSL', 'CPIMEDSL', 'CUSR0000SA0L2', 'CUSR0000SA0L5'
    ]
    
    # Use only the specified number of features
    feature_names = feature_names[:n_features]
    
    # Create target variable (inflation proxy)
    # This will be a persistent but volatile series
    inflation_base = 2.0  # 2% average inflation
    inflation_volatility = 0.5
    inflation = np.zeros(n_periods)
    inflation[0] = inflation_base + np.random.randn() * inflation_volatility
    
    # Generate autoregressive inflation series
    for i in range(1, n_periods):
        inflation[i] = (0.8 * inflation[i-1] + 
                       0.2 * inflation_base + 
                       np.random.randn() * inflation_volatility)
    
    # Create dataframe
    data = {'sasdate': dates.strftime('%Y-%m-%d')}
    
    # Add inflation as the first feature (target)
    data['CPIAUCSL'] = 100 * (1 + inflation/100).cumprod()
    
    # Generate correlated features
    for i, feature_name in enumerate(feature_names):
        if feature_name == 'CPIAUCSL':
            continue  # Already added
        
        # Create features with varying correlation to inflation
        correlation = np.random.uniform(-0.5, 0.9)
        noise_level = np.random.uniform(0.5, 2.0)
        
        # Generate correlated series
        feature = np.zeros(n_periods)
        base_value = np.random.uniform(50, 150)
        feature[0] = base_value
        
        for j in range(1, n_periods):
            trend = 0.001 * j
            seasonal = 0.05 * np.sin(2 * np.pi * j / 4)  # Quarterly seasonality
            autocorr = 0.7 * (feature[j-1] - base_value)
            inflation_effect = correlation * (inflation[j] - inflation_base)
            noise = np.random.randn() * noise_level
            
            feature[j] = base_value + trend + seasonal + autocorr + inflation_effect + noise
        
        # Add some missing values randomly (5% missing)
        missing_mask = np.random.random(n_periods) < 0.05
        feature_series = pd.Series(feature)
        feature_series[missing_mask] = np.nan
        
        data[feature_name] = feature_series
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    from pathlib import Path
    Path(save_path).mkdir(parents=True, exist_ok=True)
    filepath = Path(save_path) / filename
    df.to_csv(filepath, index=False)
    
    print(f"✓ Sample data generated: {filepath}")
    print(f"  Periods: {n_periods}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    
    return df


if __name__ == "__main__":
    print("Generating sample FRED-like data...")
    print("="*60)
    
    # Generate quarterly data
    print("\nGenerating quarterly data...")
    generate_sample_data(n_periods=200, n_features=30, frequency='Q', save_path='data/')
    
    # Generate monthly data
    print("\nGenerating monthly data...")
    generate_sample_data(n_periods=600, n_features=30, frequency='M', save_path='data/')
    
    print("\n" + "="*60)
    print("Sample data generation complete!")
    print("You can now run: python main.py")
