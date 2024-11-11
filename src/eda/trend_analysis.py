from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

def decompose_series(data, model="multiplicative", column="Close"):
    result = seasonal_decompose(data[column], model=model, period=30)
    result.plot()
    return result

def decompose_time_series(data, column='TSLA'):
    ts_decomposed = seasonal_decompose(data[column].dropna(), model='additive', period=365)
    return ts_decomposed

def adf_test(data, column='TSLA'):
    adf_test_result = adfuller(data[column].dropna())
    return adf_test_result
