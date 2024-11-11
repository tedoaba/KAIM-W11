import numpy as np

def calculate_value_at_risk(data, column='TSLA_pct_change', confidence_level=0.05):
    VaR = np.percentile(data[column].dropna(), confidence_level * 100)
    return VaR

def calculate_sharpe_ratio(data, column='TSLA_pct_change', risk_free_rate=0):
    mean_return = data[column].mean()
    std_return = data[column].std()
    sharpe_ratio = (mean_return - risk_free_rate) / std_return
    return sharpe_ratio


def calculate_sortino_ratio(data, column='TSLA_pct_change', risk_free_rate=0):
    downside_std = data[column][data[column] < 0].std()
    sortino_ratio = (data[column].mean() - risk_free_rate) / downside_std
    return sortino_ratio


# 6. Risk and Performance Metrics
def calculate_risk_metrics(data, column='TSLA', confidence_level=0.05):
    """
    Calculate Value at Risk (VaR) and Sharpe Ratio for specified asset.
    """
    # Value at Risk (VaR)
    VaR = np.percentile(data[column].pct_change().dropna(), confidence_level * 100)
    print(f"Value at Risk (VaR) at 95% confidence level for {column}: {VaR:.4f}")
    
    # Sharpe Ratio
    mean_return = data[column].pct_change().mean()
    std_return = data[column].pct_change().std()
    sharpe_ratio = mean_return / std_return
    print(f"Sharpe Ratio for {column}: {sharpe_ratio:.2f}")
    

def calculate_drawdown(data, column='TSLA'):
    """
    Calculate maximum drawdown for the specified asset.
    """
    roll_max = data[column].cummax()
    daily_drawdown = data[column] / roll_max - 1.0
    max_drawdown = daily_drawdown.cummin()
    print(f"Maximum Drawdown for {column}: {max_drawdown.min():.2f}")
    return max_drawdown.min()
    
