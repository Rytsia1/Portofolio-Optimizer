import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.optimizer import portfolio_performance

NEON_GREEN = "#a3ff00"

def plot_efficient_frontier(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    optimal_weights: pd.Series,
    risk_free_rate: float = 0.02,
    num_portfolios: int = 10000
) -> plt.Figure:
    """
    Plots the Efficient Frontier using Monte Carlo simulation on a dark background.

    Args:
        mean_returns (pd.Series): Annualized mean returns for each asset.
        cov_matrix (pd.DataFrame): Annualized covariance matrix.
        optimal_weights (pd.Series): The weights of the theoretically optimal portfolio.
        risk_free_rate (float, optional): The risk-free rate of return. Defaults to 0.02.
        num_portfolios (int, optional): Number of random portfolios to generate. Defaults to 10000.

    Returns:
        plt.Figure: Matplotlib figure object for rendering in Streamlit.
    """
    plt.style.use('dark_background')

    num_assets = len(mean_returns)
    results = np.zeros((3, num_portfolios))

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        perf_returns, perf_volatility, sharpe = portfolio_performance(
            weights, mean_returns, cov_matrix, risk_free_rate
        )
        results[0, i] = perf_volatility
        results[1, i] = perf_returns
        results[2, i] = sharpe

    opt_return, opt_volatility, opt_sharpe = portfolio_performance(
        optimal_weights.values, mean_returns, cov_matrix, risk_free_rate
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    # Transparent background to blend with Streamlit dark theme
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')

    scatter = ax.scatter(
        results[0, :],
        results[1, :],
        c=results[2, :],
        cmap='viridis',
        marker='o',
        s=8,
        alpha=0.5,
        label='Random Portfolios'
    )

    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label('Sharpe Ratio', color='white')
    colorbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(colorbar.ax.yaxis.get_ticklabels(), color='white')

    # Optimal portfolio — neon green star
    ax.scatter(
        opt_volatility,
        opt_return,
        marker='*',
        color=NEON_GREEN,
        s=500,
        zorder=5,
        label=f'Optimal Portfolio  |  Sharpe: {opt_sharpe:.2f}'
    )

    ax.set_title('Efficient Frontier — Monte Carlo Simulation', color='white', pad=14, fontsize=13)
    ax.set_xlabel('Annualized Volatility (Risk)', color='white')
    ax.set_ylabel('Annualized Expected Return', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
    ax.grid(True, linestyle='--', alpha=0.2, color='white')
    ax.legend(labelspacing=0.8, facecolor='#141414', edgecolor='#333333', labelcolor='white')

    plt.tight_layout()
    return fig
