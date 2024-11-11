from prophet import Prophet

def fit_prophet_model(data):
    model = Prophet()
    model.fit(data)
    return model

def create_future_dates(model, periods):
    return model.make_future_dataframe(periods=periods)

def generate_forecast(model, future_dates):
    return model.predict(future_dates)
