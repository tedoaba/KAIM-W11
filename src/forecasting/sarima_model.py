from statsmodels.tsa.statespace.sarimax import SARIMAX

def build_sarima_model(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    model = SARIMAX(data['Close'], order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    return model_fit

def forecast_sarima(model_fit, steps=5):
    return model_fit.forecast(steps=steps)
