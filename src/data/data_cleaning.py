import pandas as pd

def handle_missing_values(data, method="ffill"):
    return data.fillna(method=method)

def remove_outliers(data, column, threshold=3):
    mean = data[column].mean()
    std = data[column].std()
    return data[(data[column] >= mean - threshold * std) & (data[column] <= mean + threshold * std)]
