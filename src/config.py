import yaml

selected_tickers = ['TSLA', 'BND', 'SPY']
start_date = '2015-01-01'
end_date = '2024-10-31'
num_portfolios = 10000
risk_free_rate = 0.01


def load_config(config_path="../config/config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

config = load_config()
