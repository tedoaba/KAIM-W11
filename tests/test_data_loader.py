import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime

# Add `src` directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import functions from data_loader
from data.data_loader import load_dataset, load_stock_data, retrieve_stock_data, get_close_prices


class TestDataLoader(unittest.TestCase):

    @patch('data.data_loader.fetch_data')
    def test_load_dataset(self, mock_fetch_data):
        # Mock the data returned by fetch_data
        assets = ['TSLA', 'BND']
        start_date = '2023-01-01'
        end_date = '2023-12-31'
        
        # Ensure mock data has the correct number of columns (Date + len(assets))
        mock_data = pd.DataFrame({
            'Date': pd.date_range(start=start_date, periods=10),
            'TSLA': range(10),
            'BND': range(10, 20)
        })
        mock_data.set_index('Date', inplace=True)  # Set 'Date' as the index to match real data
        mock_fetch_data.return_value = mock_data

        # Call load_dataset
        result = load_dataset(assets, start_date, end_date)

        # Assert the results
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('Date', result.columns)
        self.assertEqual(result.columns.tolist(), ['Date', 'TSLA', 'BND'])  # Check column names
        self.assertEqual(len(result), 10)

    @patch('data.data_loader.yf.download')
    def test_load_stock_data(self, mock_yf_download):
        # Mock the data returned by yfinance download
        ticker = 'TSLA'
        start_date = '2023-01-01'
        end_date = '2023-12-31'
        mock_data = pd.DataFrame({
            'Date': pd.date_range(start=start_date, periods=10),
            'Adj Close': range(10)
        })
        mock_data.set_index('Date', inplace=True)
        mock_yf_download.return_value = mock_data

        # Call load_stock_data
        result = load_stock_data(ticker, start_date, end_date)

        # Assert the results
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (10, 1))
        self.assertIn('Adj Close', result.columns)

    @patch('data.data_loader.yf.download')
    def test_retrieve_stock_data(self, mock_yf_download):
        # Mock the data returned by yfinance download for multiple tickers
        tickers = ['TSLA', 'BND']
        start_date = '2023-01-01'
        end_date = '2023-12-31'
        mock_data_TSLA = pd.DataFrame({
            'Date': pd.date_range(start=start_date, periods=10),
            'Adj Close': range(10)
        }).set_index('Date')
        mock_data_BND = pd.DataFrame({
            'Date': pd.date_range(start=start_date, periods=10),
            'Adj Close': range(10, 20)
        }).set_index('Date')
        mock_yf_download.side_effect = [mock_data_TSLA, mock_data_BND]

        # Call retrieve_stock_data
        result = retrieve_stock_data(tickers, start_date, end_date)

        # Assert the results
        self.assertIsInstance(result, dict)
        self.assertIn('TSLA', result)
        self.assertIn('BND', result)
        self.assertIsInstance(result['TSLA'], pd.DataFrame)
        self.assertIsInstance(result['BND'], pd.DataFrame)

    def test_get_close_prices(self):
        # Create mock stock_data dictionary
        stock_data = {
            'TSLA': pd.DataFrame({'Adj Close': range(10)}),
            'BND': pd.DataFrame({'Adj Close': range(10, 20)})
        }

        # Call get_close_prices
        result = get_close_prices(stock_data)

        # Assert the results
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (10, 2))
        self.assertEqual(result.columns.tolist(), ['TSLA', 'BND'])


if __name__ == '__main__':
    unittest.main()
