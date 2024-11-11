import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def download_data(assets, start_date, end_date):
    data = yf.download(assets, start=start_date, end=end_date)
    data = data['Adj Close']
    data.columns = [asset for asset in assets]
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    return data

def handle_missing_values(data):
    data.fillna(method='ffill', inplace=True)
    data.interpolate(method='linear', inplace=True)
    return data

def normalize_data(data):
    scaler = MinMaxScaler()
    data[['TSLA', 'BND', 'SPY']] = scaler.fit_transform(data[['TSLA', 'BND', 'SPY']])
    return data

# 2. Data Preprocessing
def preprocess_data(data):
    """
    Handle missing values and normalize asset prices using MinMaxScaler.
    """
    data.fillna(method='ffill', inplace=True)
    data.interpolate(method='linear', inplace=True)
    
    scaler = MinMaxScaler()
    data[['TSLA', 'BND', 'SPY']] = scaler.fit_transform(data[['TSLA', 'BND', 'SPY']])
    return data


# 3. Exploratory Data Analysis (EDA)
def perform_eda(data):
    """
    Display basic statistics, check missing values, and print correlation matrix.
    """
    print("Basic Statistics:\n", data.describe())
    print("\nData Types and Missing Values:\n", data.info())
    print("\nMissing Values:\n", data.isnull().sum())
    print("\nCorrelation Matrix:\n", data[['TSLA', 'BND', 'SPY']].corr())


def fill_missing_values(data):
    print(data.isnull().sum())
    return data.fillna(method='ffill')

def extract_close_prices(data):
    return data['Adj Close']

def scale_data(data, scaler=None):
    if not scaler:
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(np.array(data).reshape(-1, 1))
    else:
        data_scaled = scaler.transform(np.array(data).reshape(-1, 1))
    return data_scaled, scaler

def rename_columns_for_prophet(data):
    data = data[['Adj Close']].reset_index()
    data.columns = ['ds', 'y']
    return data

def remove_tz_from_dataframe(df_in):
    df = df_in.copy()
    for col in df.select_dtypes(include=['datetime64[ns, UTC]']).columns:
        df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
        df[col] = df[col].dt.tz_localize(None)
    return df