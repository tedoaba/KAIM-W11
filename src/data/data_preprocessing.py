import yfinance as yf
import pandas as pd
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

