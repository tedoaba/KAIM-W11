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


def retrieve_stock_data(tickers, start_date, end_date):
    stock_data = {}
    for symbol in tickers:
        try:
            stock_data[symbol] = yf.download(symbol, start=start_date, end=end_date)
            print(f"Data successfully retrieved for {symbol}.")
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
    return stock_data

def get_close_prices(stock_data):
    close_prices = pd.concat([data['Adj Close'] for data in stock_data.values()], axis=1)
    close_prices.columns = stock_data.keys()  # Set column names as selected tickers
    return close_prices
