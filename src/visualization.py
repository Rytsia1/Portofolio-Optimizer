import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.optimizer import portfolio_performance

def plot_efficient_frontier(
    mean_returns: pd.Series, 
    cov_matrix: pd.DataFrame, 
    optimal_weights: pd.Series, 
    risk_free_rate: float = 0.02, 
    num_portfolios: int = 10000
) -> plt.Figure:
    """
    Plots the Efficient Frontier using Monte Carlo simulation.
    
    Args:
        mean_returns (pd.Series): Annualized mean returns for each asset.
        cov_matrix (pd.DataFrame): Annualized covariance matrix.
        optimal_weights (pd.Series): The weights of the theoretically optimal portfolio.
        risk_free_rate (float, optional): The risk-free rate of return. Defaults to 0.02.
        num_portfolios (int, optional): Number of random portfolios to generate. Defaults to 10000.
    """
    num_assets = len(mean_returns)
    results = np.zeros((3, num_portfolios))
    
    # Generate random portfolios
    for i in range(num_portfolios):
        # Generate random weights and normalize to sum to 1.0
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        
        # Calculate performance metrics using our optimizer's helper function
        perf_returns, perf_volatility, sharpe = portfolio_performance(
            weights, mean_returns, cov_matrix, risk_free_rate
        )
        
        results[0, i] = perf_volatility
        results[1, i] = perf_returns
        results[2, i] = sharpe
        
    # Calculate performance of the provided optimal portfolio
    opt_return, opt_volatility, opt_sharpe = portfolio_performance(
        optimal_weights.values, mean_returns, cov_matrix, risk_free_rate
    )
    
    # Create the plot
    fig = plt.figure(figsize=(10, 7))
    
    # Plot the random portfolios scatter
    scatter = plt.scatter(
        results[0, :], 
        results[1, :], 
        c=results[2, :], 
        cmap='viridis', 
        marker='o', 
        s=10, 
        alpha=0.4,
        label='Random Portfolios'
    )
    
    # Add the continuous colorbar mapped to the Sharpe Ratios
    colorbar = plt.colorbar(scatter)
    colorbar.set_label('Sharpe Ratio')
    
    # Plot the optimal portfolio heavily superimposed
    plt.scatter(
        opt_volatility, 
        opt_return, 
        marker='*', 
        color='red', 
        s=400, 
        label=f'Optimal Portfolio\nSharpe: {opt_sharpe:.2f}'
    )
    
    # Add axis metadata
    plt.title('Efficient Frontier (Monte Carlo Simulation)')
    plt.xlabel('Annualized Volatility (Risk)')
    plt.ylabel('Annualized Expected Return')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(labelspacing=0.8)
    
    plt.tight_layout()
    return fig
