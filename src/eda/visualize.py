import matplotlib.pyplot as plt

def plot_close_price(data):
    # Check for NaT values in the 'Date' column
    if data['Date'].isnull().any():
        print("Warning: There are missing values in the 'Date' column.")
    # Plot close price
    plt.plot(data['Date'], data['Close'], label="Close Price")
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()


def plot_volatility(data, window=30):
    data['Volatility'] = data['Close'].rolling(window=window).std()
    plt.figure(figsize=(10, 6))
    plt.plot(data['Date'], data['Volatility'], label="Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.legend()
    plt.show()
