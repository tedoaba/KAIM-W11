import os
import sys

sys.path.append(os.path.abspath('../src'))

import pandas as pd
from utils.yfinance_loader import fetch_data

def load_data(ticker, start_date, end_date):
    data = fetch_data(ticker, start_date, end_date)
    data['Date'] = pd.to_datetime(data.index)  # Convert the index to a 'Date' column
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')  # Convert to datetime, invalid dates become NaT
    data.reset_index(drop=True, inplace=True)  # Reset the index to avoid ambiguity
    data = data.sort_values(by='Date')  # Sort by the 'Date' column
    return data
