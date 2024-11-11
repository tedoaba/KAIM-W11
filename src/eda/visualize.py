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
    plt.show()

def plot_correlation_matrix(data):
    correlation_matrix = data[['TSLA', 'BND', 'SPY']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Correlation Matrix of Asset Prices")
    plt.show()
