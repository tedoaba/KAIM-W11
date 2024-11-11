import os
import sys
import yfinance as yf
import pandas as pd
from src.utils.yfinance_loader import fetch_data


sys.path.append(os.path.abspath('../src'))

def load_dataset(assets, start_date, end_date):
    """
    Load adjusted closing prices from Yahoo Finance for specified assets and date range.
    """

    data = fetch_data(assets, start_date=start_date, end_date=end_date)
    data.columns = assets
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    return data

def load_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data
