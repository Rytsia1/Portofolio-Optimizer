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
    plt.style.use('default')

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

    # White background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

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
    colorbar.set_label('Sharpe Ratio', color='black')
    colorbar.ax.yaxis.set_tick_params(color='black')
    plt.setp(colorbar.ax.yaxis.get_ticklabels(), color='black')

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

    ax.set_title('Efficient Frontier — Monte Carlo Simulation', color='black', pad=14, fontsize=13)
    ax.set_xlabel('Annualized Volatility (Risk)', color='black')
    ax.set_ylabel('Annualized Expected Return', color='black')
    ax.tick_params(colors='black')
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')
    ax.grid(True, linestyle='--', alpha=0.4, color='grey')
    ax.legend(labelspacing=0.8, facecolor='white', edgecolor='#cccccc', labelcolor='black')

    plt.tight_layout()
    return fig
