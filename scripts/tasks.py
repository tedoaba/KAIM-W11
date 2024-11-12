import os
import sys
import numpy as np
import tensorflow as tf
import warnings


sys.path.append(os.path.abspath('..'))

from src.eda.trend_analysis import decompose_time_series, adf_test
from src.data.data_loader import load_dataset, load_stock_data, retrieve_stock_data, get_close_prices
from src.data.data_preprocessing import preprocess_data, perform_eda, fill_missing_values, extract_close_prices, scale_data, rename_columns_for_prophet, remove_tz_from_dataframe
from src.eda.visualize import plot_normalized_prices, plot_percentage_change, decompose_time_series, plot_moving_average_crossover, plot_rolling_stats, plot_correlation_matrix, plot_stock_data, plot_train_test_split, plot_forecast, plot_forecast_prophet, plot_trend_volatility, plot_opportunities_and_risks, plot_missing_values_heatmap, plot_log_returns, plot_volatility, plot_portfolios
from src.portfolio.risk_metrics import calculate_risk_metrics, calculate_drawdown

from src.data.train_test_split import split_data
from src.forecasting.arima_model import find_optimal_arima_order, train_arima, arima_forecast, fit_arima_model
from src.forecasting.sarima_model import find_optimal_sarima_order, train_sarima, sarima_forecast
from src.forecasting.lstm_model import prepare_lstm_data, build_lstm_model, train_lstm, lstm_forecast
from src.forecasting.model_evaluation import calculate_metrics

from src.forecasting.prophet_model import fit_prophet_model, create_future_dates, generate_forecast
from src.utils.time_utils import calculate_volatility, identify_opportunity_periods, identify_risk_periods

from src.eda.price_analysis import calculate_log_returns, calculate_variance, calculate_volatility, calculate_covariance, calculate_missing_values, calculate_expected_return, calculate_portfolio_variance
from src.portfolio.performance_metrics import simulate_random_portfolios
import src.config as cfg



def task_1():
    
    # Load Data
    data = load_dataset(cfg.selected_tickers, cfg.start_date, cfg.end_date)
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
    data = load_stock_data(cfg.tsla_ticker, cfg.start_date, cfg.end_date)
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
    inputs = scaler.transform(np.array(train.iloc[-60:, 0].tolist() + test.iloc[:, 0].tolist()).reshape(-1, 1))
    X_test = [inputs[i - 60:i] for i in range(60, len(inputs))]
    X_test = np.array(X_test)
    lstm_forecast_vals = lstm_forecast(lstm_model, X_test, scaler)
    plot_forecast(test, lstm_forecast_vals, title="LSTM Model Forecast vs Actual")

    # Calculate evaluation metrics
    print("ARIMA Metrics:", calculate_metrics(test, arima_forecast_vals))
    print("SARIMA Metrics:", calculate_metrics(test, sarima_forecast_vals))
    print("LSTM Metrics:", calculate_metrics(test, lstm_forecast_vals))


def task_3():

    tf.random.set_seed(42)
    np.random.seed(42)
    warnings.filterwarnings('ignore')

    tsla_data = load_stock_data(cfg.tsla_ticker, cfg.start_date, cfg.end_date)
    tsla_data = fill_missing_values(tsla_data)
    tsla_data = rename_columns_for_prophet(tsla_data)
    tsla_data = remove_tz_from_dataframe(tsla_data)

    # ARIMA modeling
    arima_fit = fit_arima_model(tsla_data['y'])
    print(arima_fit.summary())

    # Prophet modeling
    prophet_model = fit_prophet_model(tsla_data)
    future_dates = create_future_dates(prophet_model, periods=cfg.forecast_horizon)
    forecast = generate_forecast(prophet_model, future_dates)

    # Plot results
    plot_forecast_prophet(tsla_data, forecast, cfg.tsla_ticker, cfg.forecast_horizon)

    # Analyze trend and volatility
    trend = forecast['yhat']
    trend_diff = trend.diff()
    plot_trend_volatility(trend, forecast)

    volatility = calculate_volatility(trend_diff)
    print(f"Forecasted Trend Volatility: {volatility}")

    # Identify opportunities and risks
    opportunity_periods = identify_opportunity_periods(forecast, trend_diff, volatility)
    risk_periods = identify_risk_periods(forecast, trend_diff, volatility)
    plot_opportunities_and_risks(forecast, opportunity_periods, risk_periods)

    # Summary statistics
    print(f"Potential Opportunity Periods:\n{opportunity_periods[['ds', 'yhat', 'yhat_upper']]}")
    print(f"\nPotential Risk Periods:\n{risk_periods[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]}")


def task_4():
    # Load Data
    stock_data = retrieve_stock_data(cfg.selected_tickers, cfg.start_date, cfg.end_date)
    close_prices = get_close_prices(stock_data)

    # Missing Values Analysis
    missing_values = calculate_missing_values(close_prices)
    plot_missing_values_heatmap(missing_values)

    # Calculate Log Returns
    log_returns = calculate_log_returns(close_prices)
    print(log_returns.head())

    # Variance and Volatility
    variance = calculate_variance(log_returns)
    volatility = calculate_volatility(variance)
    plot_volatility(volatility)

    # Portfolio Simulation
    ind_er = log_returns.mean()
    cov_matrix = calculate_covariance(log_returns)
    portfolios = simulate_random_portfolios(cfg.num_portfolios, ind_er, cov_matrix, cfg.selected_tickers)

    # Plot Efficient Frontier
    min_vol_port = portfolios.iloc[portfolios['Volatility'].idxmin()]
    optimal_risky_port = portfolios.iloc[((portfolios['Returns'] - cfg.risk_free_rate) / portfolios['Volatility']).idxmax()]
    plot_portfolios(portfolios, min_vol_port, optimal_risky_port)

