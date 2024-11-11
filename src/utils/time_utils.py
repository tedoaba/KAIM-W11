import pandas as pd

def calculate_volatility(trend_diff):
    return trend_diff.std()

def identify_opportunity_periods(forecast, trend_diff, volatility):
    return forecast[(trend_diff > 0) & (forecast['yhat'] > forecast['yhat'].rolling(20).mean())]

def identify_risk_periods(forecast, trend_diff, volatility):
    return forecast[(trend_diff < 0) | (forecast['yhat_upper'] - forecast['yhat'] > volatility)]
