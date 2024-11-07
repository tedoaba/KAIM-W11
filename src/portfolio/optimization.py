import numpy as np
from scipy.optimize import minimize

def portfolio_optimization(returns):
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    def objective(weights):
        return -np.dot(weights, mean_returns) / np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for _ in range(len(mean_returns)))
    result = minimize(objective, len(mean_returns) * [1. / len(mean_returns)], bounds=bounds, constraints=constraints)
    return result.x
