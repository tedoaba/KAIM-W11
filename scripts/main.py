# scripts/main.py

import os
import sys

sys.path.append(os.path.abspath('..'))

from src.config import load_config
from src.utils.yfinance_loader import fetch_data
from src.data.data_loader import load_data
from src.data.data_cleaning import handle_missing_values, remove_outliers
from src.eda.visualize import plot_close_price, plot_volatility

def main():
    # Load configuration from config.yaml
    config = load_config()
    ticker = config['general']['ticker']
    start_date = config['general']['start_date']
    end_date = config['general']['end_date']

    # Download stock data
    data = load_data(ticker, start_date, end_date)

    # Data cleaning
    data = handle_missing_values(data)
    data = remove_outliers(data, 'Close')

    # Visualization
    plot_close_price(data)
    plot_volatility(data)

    print("End")

if __name__ == "__main__":
    main()
