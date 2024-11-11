from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

def find_optimal_sarima_order(train_data):
    model = auto_arima(train_data, seasonal=True, m=12, trace=True)
    return model.order, model.seasonal_order

def train_sarima(train_data, order, seasonal_order):
    model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    return model_fit

def sarima_forecast(model_fit, steps):
    forecast = model_fit.forecast(steps=steps)
    return forecast
