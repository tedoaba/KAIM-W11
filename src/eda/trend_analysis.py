from statsmodels.tsa.seasonal import seasonal_decompose

def decompose_series(data, model="multiplicative", column="Close"):
    result = seasonal_decompose(data[column], model=model, period=30)
    result.plot()
    return result
