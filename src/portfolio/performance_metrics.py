import numpy as np
import pandas as pd

def simulate_random_portfolios(num_portfolios, ind_er, cov_matrix, tickers):
    num_assets = len(tickers)
    p_ret, p_vol, p_weights = [], [], []

    for _ in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)  # Normalize weights to sum up to 1
        p_weights.append(weights)
        
        # Portfolio returns
        p_ret.append(np.dot(weights, ind_er))
        
        # Portfolio variance and volatility
        var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()
        sd = np.sqrt(var)
        ann_sd = sd * np.sqrt(250)  # Annual standard deviation
        p_vol.append(ann_sd)

    portfolios = pd.DataFrame({
        'Returns': p_ret,
        'Volatility': p_vol,
        **{f"{symbol} weight": [w[i] for w in p_weights] for i, symbol in enumerate(tickers)}
    })
    return portfolios
