import os
import sys

sys.path.append(os.path.abspath('..'))

import unittest
from src.portfolio.risk_metrics import calculate_sharpe_ratio, calculate_var
from src.portfolio.optimization import portfolio_optimization
import numpy as np
import pandas as pd

class TestPortfolio(unittest.TestCase):
    def setUp(self):
        self.returns = np.array([0.01, 0.02, 0.015, -0.005, 0.03])

    def test_calculate_sharpe_ratio(self):
        sharpe_ratio = calculate_sharpe_ratio(self.returns)
        self.assertIsInstance(sharpe_ratio, float)

    def test_calculate_var(self):
        var = calculate_var(self.returns, confidence_level=0.95)
        self.assertIsInstance(var, float)

    def test_portfolio_optimization(self):
        data = pd.DataFrame({'A': [0.01, 0.02, 0.03], 'B': [0.02, 0.03, 0.04]})
        weights = portfolio_optimization(data)
        self.assertAlmostEqual(weights.sum(), 1.0, places=2)

if __name__ == '__main__':
    unittest.main()
