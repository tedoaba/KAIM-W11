# scripts/main.py

import os
import sys

sys.path.append(os.path.abspath('..'))

from src.data.data_preprocessing import download_data, handle_missing_values, normalize_data
from src.eda.trend_analysis import decompose_time_series, adf_test
from src.data.data_loader import load_dataset
from src.data.data_preprocessing import preprocess_data, perform_eda
from src.eda.visualize import plot_normalized_prices, plot_percentage_change, decompose_time_series, plot_moving_average_crossover, plot_rolling_stats, plot_correlation_matrix
from src.portfolio.risk_metrics import calculate_risk_metrics, calculate_drawdown


if __name__ == "__main__":
    # Define assets, date range, and load data
    assets = ['TSLA', 'BND', 'SPY']
    start_date = '2020-01-01'
    end_date = '2023-01-01'
    
    # Load and preprocess data
    data = load_dataset(assets, start_date, end_date)
    data = preprocess_data(data)
    
    # Perform EDA
    perform_eda(data)
    
    # Visualization
    plot_normalized_prices(data)
    plot_percentage_change(data)
    plot_rolling_stats(data)
    plot_correlation_matrix(data)
    
    # Time series decomposition
    decompose_time_series(data, column='TSLA')
    
    # Risk metrics
    calculate_risk_metrics(data, column='TSLA')
    calculate_drawdown(data, column='TSLA')
    
    # Moving average crossover
    plot_moving_average_crossover(data, column='TSLA')
