import os
import sys

sys.path.append(os.path.abspath('../src'))

import unittest
from eda.trend_analysis import decompose_series
import pandas as pd

class TestEDA(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            'Date': pd.date_range(start="2022-01-01", periods=100, freq="D"),
            'Close': [100 + i * 0.5 for i in range(100)]
        }).set_index("Date")

    def test_decompose_series(self):
        result = decompose_series(self.data, column="Close")
        self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()
