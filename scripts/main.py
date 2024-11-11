# scripts/main.py

import os
import sys
import numpy as np

sys.path.append(os.path.abspath('..'))

from src.eda.trend_analysis import decompose_time_series, adf_test
from src.data.data_loader import load_dataset, load_stock_data
from src.data.data_preprocessing import preprocess_data, perform_eda, fill_missing_values, extract_close_prices, scale_data
from src.eda.visualize import plot_normalized_prices, plot_percentage_change, decompose_time_series, plot_moving_average_crossover, plot_rolling_stats, plot_correlation_matrix, plot_stock_data, plot_train_test_split, plot_forecast
from src.portfolio.risk_metrics import calculate_risk_metrics, calculate_drawdown


from src.data.train_test_split import split_data
from src.forecasting.arima_model import find_optimal_arima_order, train_arima, arima_forecast
from src.forecasting.sarima_model import find_optimal_sarima_order, train_sarima, sarima_forecast
from src.forecasting.lstm_model import prepare_lstm_data, build_lstm_model, train_lstm, lstm_forecast
from src.forecasting.model_evaluation import calculate_metrics


def task_1():
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


def task_2():
    
    # Load data
    ticker = 'TSLA'
    start_date = '2015-01-01'
    end_date = '2024-10-31'
    data = load_stock_data(ticker, start_date, end_date)
    data = fill_missing_values(data)
    close_prices = extract_close_prices(data)

    # Split data
    train, test = split_data(close_prices)
    plot_train_test_split(train, test)

    # ARIMA model
    arima_order = find_optimal_arima_order(train)
    arima_model = train_arima(train, arima_order)
    arima_forecast_vals = arima_forecast(arima_model, steps=len(test))
    plot_forecast(test, arima_forecast_vals, title="ARIMA Model Forecast vs Actual")

    # SARIMA model
    sarima_order, sarima_seasonal_order = find_optimal_sarima_order(train)
    sarima_model = train_sarima(train, sarima_order, sarima_seasonal_order)
    sarima_forecast_vals = sarima_forecast(sarima_model, steps=len(test))
    plot_forecast(test, sarima_forecast_vals, title="SARIMA Model Forecast vs Actual")

    # LSTM model
    scaler = None
    train_scaled, scaler = scale_data(train, scaler)
    X_train, y_train = prepare_lstm_data(train_scaled)
    lstm_model = build_lstm_model((X_train.shape[1], 1))
    lstm_model = train_lstm(lstm_model, X_train, y_train)

    # LSTM forecasting
    inputs = scaler.transform(np.array(train[-60:].tolist() + test.tolist()).reshape(-1, 1))
    X_test = [inputs[i - 60:i] for i in range(60, len(inputs))]
    X_test = np.array(X_test)
    lstm_forecast_vals = lstm_forecast(lstm_model, X_test, scaler)
    plot_forecast(test, lstm_forecast_vals, title="LSTM Model Forecast vs Actual")

    # Calculate evaluation metrics
    print("ARIMA Metrics:", calculate_metrics(test, arima_forecast_vals))
    print("SARIMA Metrics:", calculate_metrics(test, sarima_forecast_vals))
    print("LSTM Metrics:", calculate_metrics(test, lstm_forecast_vals))



if __name__ == "__main__":
    task_1()
    task_2()