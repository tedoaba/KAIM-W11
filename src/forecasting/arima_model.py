from statsmodels.tsa.arima.model import ARIMA

def build_arima_model(data, order=(1, 1, 1)):
    model = ARIMA(data['Close'], order=order)
    model_fit = model.fit()
    return model_fit

def forecast_arima(model_fit, steps=5):
    return model_fit.forecast(steps=steps)
