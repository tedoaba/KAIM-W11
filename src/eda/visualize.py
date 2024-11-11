import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose


def plot_normalized_prices(data):
    """
    Plot normalized closing prices over time for each asset.
    """
    plt.figure(figsize=(14, 7))
    for asset in ['TSLA', 'BND', 'SPY']:
        plt.plot(data['Date'], data[asset], label=f'{asset} (Normalized)')
    plt.title('Normalized Closing Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.savefig('../figures/normalized_price.png', format='png', dpi=300)
    plt.show()


def plot_percentage_change(data):
    """
    Plot daily percentage change in asset prices.
    """
    data[['TSLA_pct_change', 'BND_pct_change', 'SPY_pct_change']] = data[['TSLA', 'BND', 'SPY']].pct_change()
    plt.figure(figsize=(14, 7))
    for asset in ['TSLA_pct_change', 'BND_pct_change', 'SPY_pct_change']:
        plt.plot(data['Date'], data[asset], label=f'{asset} Daily % Change', alpha=0.7)
    plt.title('Daily Percentage Change in Asset Prices')
    plt.xlabel('Date')
    plt.ylabel('Daily % Change')
    plt.legend()
    plt.savefig('../figures/daily_percentage_change.png', format='png', dpi=300)
    plt.show()


# 5. Time Series Analysis
def decompose_time_series(data, column='TSLA'):
    """
    Perform seasonal decomposition on the specified time series column.
    """
    decomposed = seasonal_decompose(data[column].dropna(), model='additive', period=365)
    plt.figure(figsize=(14, 10))
    plt.subplot(411)
    plt.plot(decomposed.observed, label='Observed')
    plt.legend(loc='upper left')

    plt.subplot(412)
    plt.plot(decomposed.trend, label='Trend', color='orange')
    plt.legend(loc='upper left')

    plt.subplot(413)
    plt.plot(decomposed.seasonal, label='Seasonal', color='green')
    plt.legend(loc='upper left')

    plt.subplot(414)
    plt.plot(decomposed.resid, label='Residual', color='red')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig('../figures/decomposed_time_series.png', format='png', dpi=300)
    plt.show()


# 7. Moving Average Crossover for Buy/Sell Signals
def plot_moving_average_crossover(data, column='TSLA', short_window=50, long_window=200):
    """
    Plot moving average crossover signals for buy and sell actions.
    """
    data[f'{column}_50_MA'] = data[column].rolling(window=short_window).mean()
    data[f'{column}_200_MA'] = data[column].rolling(window=long_window).mean()
    
    plt.figure(figsize=(14, 7))
    plt.plot(data['Date'], data[column], label=f'{column}', color='blue', alpha=0.6)
    plt.plot(data['Date'], data[f'{column}_50_MA'], label=f'{short_window}-Day MA', color='red')
    plt.plot(data['Date'], data[f'{column}_200_MA'], label=f'{long_window}-Day MA', color='green')
    
    buy_signals = data[(data[f'{column}_50_MA'] > data[f'{column}_200_MA']) & (data[f'{column}_50_MA'].shift(1) <= data[f'{column}_200_MA'].shift(1))]
    sell_signals = data[(data[f'{column}_50_MA'] < data[f'{column}_200_MA']) & (data[f'{column}_50_MA'].shift(1) >= data[f'{column}_200_MA'].shift(1))]

    plt.scatter(buy_signals['Date'], buy_signals[column], label='Buy Signal', marker='^', color='green', lw=3)
    plt.scatter(sell_signals['Date'], sell_signals[column], label='Sell Signal', marker='v', color='red', lw=3)
    
    plt.title(f'{column} Price with Moving Average Crossovers')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('../figures/moving_average_crossover.png', format='png', dpi=300)
    plt.show()

def plot_rolling_stats(data):

    # Calculate rolling mean and standard deviation (30-day window)
    data['TSLA_rolling_mean'] = data['TSLA'].rolling(window=30).mean()
    data['TSLA_rolling_std'] = data['TSLA'].rolling(window=30).std()

    plt.figure(figsize=(14, 7))
    plt.plot(data['Date'], data['TSLA'], label='TSLA', color='blue', alpha=0.6)
    plt.plot(data['Date'], data['TSLA_rolling_mean'], label='TSLA Rolling Mean (30 days)', color='red')
    plt.plot(data['Date'], data['TSLA_rolling_std'], label='TSLA Rolling Std (30 days)', color='green')
    plt.title('Rolling Mean and Standard Deviation for Tesla (TSLA)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('../figures/rolling_stats.png', format='png', dpi=300)
    plt.show()

def plot_correlation_matrix(data):
    correlation_matrix = data[['TSLA', 'BND', 'SPY']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Correlation Matrix of Asset Prices")
    plt.savefig('../figures/correlation_matrix.png', format='png', dpi=300)
    plt.show()

def plot_stock_data(data, title="Stock Closing Price"):
    data.plot(figsize=(12, 6), title=title, xlabel='Date', ylabel='Price')
    plt.savefig('../figures/stock_price.png', format='png', dpi=300)
    plt.show()

def plot_train_test_split(train, test):
    plt.figure(figsize=(12, 6))
    plt.plot(train, label='Train Data')
    plt.plot(test, label='Test Data')
    plt.legend()
    plt.title("Train-Test Split")
    plt.savefig('../figures/train_test_split.png', format='png', dpi=300)
    plt.show()

def plot_forecast(actual, forecast, title="Model Forecast vs Actual"):
    plt.figure(figsize=(12, 6))
    plt.plot(actual.index, actual.values, label='Actual Prices')
    plt.plot(actual.index, forecast, label='Forecast', color='orange')
    plt.legend()
    plt.title(title)
    plt.savefig('../figures/forecast.png', format='png', dpi=300)
    plt.show()


def plot_forecast_prophet(data, forecast, ticker, forecast_horizon):
    plt.figure(figsize=(14, 7))
    plt.plot(data['ds'], data['y'], label="Historical Data")
    plt.plot(forecast['ds'], forecast['yhat'], label="Forecast", color='orange')
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='orange', alpha=0.3, label="Confidence Interval")
    plt.title(f"{ticker} Stock Price Forecast for Next {forecast_horizon // 30} Months")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.savefig('../figures/forecast_prophet.png', format='png', dpi=300)
    plt.show()

def plot_trend_volatility(trend, forecast):
    plt.figure(figsize=(12, 6))
    plt.plot(trend, label="Forecasted Trend")
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='orange', alpha=0.3, label="Volatility (Confidence Interval)")
    plt.xlabel("Date")
    plt.ylabel("Trend Value")
    plt.title("Trend and Volatility Analysis of Forecasted TSLA Stock Prices")
    plt.legend()
    plt.savefig('../figures/trend_volatility.png', format='png', dpi=300)
    plt.show()

def plot_opportunities_and_risks(forecast, opportunity_periods, risk_periods):
    plt.figure(figsize=(14, 6))
    plt.plot(forecast['ds'], forecast['yhat'], label="Forecast", color='orange')
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='orange', alpha=0.3)
    plt.scatter(opportunity_periods['ds'], opportunity_periods['yhat'], color='green', label="Opportunity", marker='^')
    plt.scatter(risk_periods['ds'], risk_periods['yhat'], color='red', label="Risk", marker='v')
    plt.title("Identified Opportunities and Risks in Forecasted TSLA Stock Prices")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.savefig('../figures/opportunities_and_risks.png', format='png', dpi=300)
    plt.show()


def plot_missing_values_heatmap(data, title='Missing Values'):
    plt.figure(figsize=(10, 6))
    sns.heatmap(data, cmap='coolwarm', annot=True, fmt='0.0f')
    plt.title(title)
    plt.savefig('../figures/missin_value_heatmap.png', format='png', dpi=300)
    plt.show()

def plot_log_returns(log_returns, title='Log Returns'):
    for symbol in log_returns.columns:
        plt.figure(figsize=(10, 6))
        plt.bar(log_returns.index, log_returns[symbol], label=symbol)
        plt.title(f'{title} for {symbol}')
        plt.xlabel('Date')
        plt.ylabel('Log Returns')
        plt.grid(True)
        plt.savefig(f'../figures/log_returns_{symbol}.png', format='png', dpi=300)
        plt.show()

def plot_volatility(volatility, title='Volatility'):
    plt.figure(figsize=(10, 6))
    plt.bar(volatility.index, volatility.values, color='#283149')
    plt.title(title)
    plt.xlabel('Stock Symbol')
    plt.ylabel('Volatility')
    plt.savefig('../figures/volatility.png', format='png', dpi=300)
    plt.show()

def plot_portfolios(portfolios, min_vol_port, optimal_risky_port):
    plt.subplots(figsize=(10, 10))
    plt.scatter(portfolios['Volatility'], portfolios['Returns'], marker='o', s=10, alpha=0.3, label='Portfolios')
    plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=500, label='Minimum Volatility Portfolio')
    plt.scatter(optimal_risky_port[1], optimal_risky_port[0], color='g', marker='*', s=500, label='Optimal Risky Portfolio')
    plt.xlabel('Volatility')
    plt.ylabel('Returns')
    plt.legend()
    plt.savefig('../figures/portfolio.png', format='png', dpi=300)
    plt.show()
