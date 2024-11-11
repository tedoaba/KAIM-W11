from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

def find_optimal_arima_order(train_data):
    model = auto_arima(train_data, seasonal=False, trace=True)
    return model.order

def train_arima(train_data, order):
    model = ARIMA(train_data, order=order)
    model_fit = model.fit()
    return model_fit

def arima_forecast(model_fit, steps):
    forecast = model_fit.forecast(steps=steps)
    return forecast

def fit_arima_model(data, order=(5,1,0)):
    model = ARIMA(data, order=order)
    return model.fit()