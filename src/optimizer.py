import numpy as np
import pandas as pd
import scipy.optimize as sco
from typing import Tuple, Dict, Any

def portfolio_performance(
    weights: np.ndarray, 
    mean_returns: pd.Series, 
    cov_matrix: pd.DataFrame, 
    risk_free_rate: float = 0.02
) -> Tuple[float, float, float]:
    """
    Calculates the expected return, volatility, and Sharpe ratio of a portfolio.

    Args:
        weights (np.ndarray): Array of asset weights in the portfolio.
        mean_returns (pd.Series): Annualized mean returns for each asset.
        cov_matrix (pd.DataFrame): Annualized covariance matrix.
        risk_free_rate (float): The risk-free rate of return. Defaults to 0.02.

    Returns:
        Tuple[float, float, float]: A tuple containing expected return, expected volatility, and Sharpe ratio.
    """
    returns = np.sum(mean_returns * weights)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (returns - risk_free_rate) / volatility
    return returns, volatility, sharpe_ratio

def negative_sharpe_ratio(
    weights: np.ndarray, 
    mean_returns: pd.Series, 
    cov_matrix: pd.DataFrame, 
    risk_free_rate: float = 0.02
) -> float:
    """
    Calculates the negative Sharpe ratio (objective function for minimization).

    Args:
        weights (np.ndarray): Array of asset weights in the portfolio.
        mean_returns (pd.Series): Annualized mean returns for each asset.
        cov_matrix (pd.DataFrame): Annualized covariance matrix.
        risk_free_rate (float): The risk-free rate of return. Defaults to 0.02.

    Returns:
        float: Negative Sharpe ratio.
    """
    return -portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2]

def optimize_portfolio(
    mean_returns: pd.Series, 
    cov_matrix: pd.DataFrame, 
    risk_free_rate: float = 0.02,
    max_weight: float = 1.0
) -> Dict[str, Any]:
    """
    Finds the optimal portfolio weights that maximize the Sharpe ratio.

    Args:
        mean_returns (pd.Series): Annualized mean returns for each asset.
        cov_matrix (pd.DataFrame): Annualized covariance matrix.
        risk_free_rate (float): The risk-free rate of return. Defaults to 0.02.
        max_weight (float): Maximum allowable weight for any single asset (0.0–1.0).
                            Defaults to 1.0 (no cap / unconstrained).

    Returns:
        Dict[str, Any]: A dictionary containing the optimal weights, expected return, 
                        volatility, and maximum Sharpe ratio.
    """
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    
    # Constraint: sum of all weights must equal 1.0
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
    
    # Bounds: each weight must be between 0.0 and max_weight (no short selling)
    bounds = tuple((0.0, max_weight) for _ in range(num_assets))
    
    # Initial guess: equal distribution of weights
    initial_guess = np.array([1.0 / num_assets] * num_assets)
    
    # Run the optimization using SciPy's Sequential Least Squares Programming (SLSQP)
    opt_result = sco.minimize(
        negative_sharpe_ratio, 
        initial_guess, 
        args=args, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints
    )
    
    # Extract optimal weights and calculate resulting performance
    optimal_weights = opt_result.x
    expected_return, volatility, max_sharpe_ratio = portfolio_performance(
        optimal_weights, mean_returns, cov_matrix, risk_free_rate
    )
    
    # Format the optimum weights as a pandas Series for clarity (assuming mean_returns was a Series)
    asset_weights = pd.Series(optimal_weights, index=mean_returns.index)
    
    return {
        'optimal_weights': asset_weights,
        'expected_return': expected_return,
        'volatility': volatility,
        'max_sharpe_ratio': max_sharpe_ratio
    }
