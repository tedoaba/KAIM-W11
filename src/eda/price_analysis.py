import numpy as np
import pandas as pd

def calculate_missing_values(data):
    return data.isna().sum().to_frame().T

def calculate_log_returns(data):
    return np.log(1 + data.pct_change())

def calculate_variance(data):
    return data.var()

def calculate_volatility(variance, days=250):
    return np.sqrt(variance * days)

def calculate_covariance(data):
    return data.cov()

def calculate_expected_return(data, weights):
    return np.dot(data.mean(), weights)

def calculate_portfolio_variance(cov_matrix, weights):
    return (cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum())
