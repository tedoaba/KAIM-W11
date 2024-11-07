from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def calculate_rmse(true_values, predictions):
    return np.sqrt(mean_squared_error(true_values, predictions))

def calculate_mae(true_values, predictions):
    return mean_absolute_error(true_values, predictions)

def calculate_mape(true_values, predictions):
    return np.mean(np.abs((true_values - predictions) / true_values)) * 100
