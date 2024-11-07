import os
import sys

sys.path.append(os.path.abspath('../src'))

import unittest
from data.data_loader import load_data
from data.data_cleaning import handle_missing_values, remove_outliers
from data.data_transformation import scale_data
import pandas as pd
import numpy as np

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # Sample data
        self.data = pd.DataFrame({
            'Close': [100, 105, np.nan, 110, 115, 120],
            'Volume': [200, 210, 220, np.nan, 230, 240]
        })

    def test_load_data(self):
        data = load_data("AAPL", "2022-01-01", "2022-02-01")
        self.assertFalse(data.empty)

    def test_handle_missing_values(self):
        cleaned_data = handle_missing_values(self.data)
        self.assertFalse(cleaned_data.isnull().values.any())

    def test_remove_outliers(self):
        data_no_outliers = remove_outliers(self.data, "Close")
        self.assertTrue(data_no_outliers.shape[0] <= self.data.shape[0])

    def test_scale_data(self):
        scaled_data, scaler = scale_data(self.data.copy(), ["Close", "Volume"])
        self.assertEqual(scaled_data.shape, self.data.shape)

if __name__ == '__main__':
    unittest.main()
