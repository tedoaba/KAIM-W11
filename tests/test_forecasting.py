import os
import sys

sys.path.append(os.path.abspath('../src'))

import unittest
from forecasting.arima_model import build_arima_model, forecast_arima
from forecasting.sarima_model import build_sarima_model, forecast_sarima
import pandas as pd

class TestForecasting(unittest.TestCase):
    def setUp(self):
        # Sample data for time series
        self.data = pd.DataFrame({'Close': [100, 101, 102, 103, 104, 105]})

    def test_arima_model(self):
        model_fit = build_arima_model(self.data, order=(1, 1, 1))
        self.assertIsNotNone(model_fit)

    def test_forecast_arima(self):
        model_fit = build_arima_model(self.data, order=(1, 1, 1))
        forecast = forecast_arima(model_fit, steps=3)
        self.assertEqual(len(forecast), 3)

    def test_sarima_model(self):
        model_fit = build_sarima_model(self.data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
        self.assertIsNotNone(model_fit)

if __name__ == '__main__':
    unittest.main()
