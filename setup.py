from setuptools import setup, find_packages

setup(
    name="forecasting_portfolio_analysis",
    version="1.0.0",
    description="A project for time series forecasting and portfolio optimization using machine learning models.",
    author="Your Name",
    author_email="your_email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy",
        "yfinance",
        "matplotlib",
        "scikit-learn",
        "statsmodels",
        "tensorflow",
        "pyyaml",
        "scipy"
    ],
    entry_points={
        "console_scripts": [
            "run_arima=src.forecasting.arima_model:main",
            "run_lstm=src.forecasting.lstm_model:main",
            "optimize_portfolio=src.portfolio.optimization:main"
        ]
    },
)
