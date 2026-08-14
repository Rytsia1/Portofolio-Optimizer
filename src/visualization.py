import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.optimizer import portfolio_performance


def plot_efficient_frontier(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    optimal_weights: pd.Series,
    risk_free_rate: float = 0.02,
    num_portfolios: int = 10000,
    theme: str = "Dark"
) -> plt.Figure:
    """
    Plots the Efficient Frontier using Monte Carlo simulation.

    Args:
        mean_returns (pd.Series): Annualized mean returns for each asset.
        cov_matrix (pd.DataFrame): Annualized covariance matrix.
        optimal_weights (pd.Series): Optimal portfolio weights.
        risk_free_rate (float): Risk-free rate. Defaults to 0.02.
        num_portfolios (int): Number of random portfolios. Defaults to 10000.
        theme (str): "Dark" or "Light" — controls matplotlib style and marker colors.

    Returns:
        plt.Figure: Matplotlib figure ready for st.pyplot().
    """
    num_assets = len(mean_returns)
    results = np.zeros((3, num_portfolios))

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        ret, vol, sharpe = portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
        results[0, i] = vol
        results[1, i] = ret
        results[2, i] = sharpe

    opt_return, opt_volatility, opt_sharpe = portfolio_performance(
        optimal_weights.values, mean_returns, cov_matrix, risk_free_rate
    )

    # Theme-specific settings
    if theme == "Dark":
        style        = 'dark_background'
        cmap         = 'plasma'
        marker_color = '#a3ff00'
        text_color   = 'white'
        grid_color   = 'white'
        grid_alpha   = 0.15
        legend_face  = '#1a1a1a'
        spine_color  = '#333333'
    else:
        style        = 'default'
        cmap         = 'viridis'
        marker_color = '#dc2626'
        text_color   = 'black'
        grid_color   = 'grey'
        grid_alpha   = 0.35
        legend_face  = 'white'
        spine_color  = '#cccccc'

    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')

        scatter = ax.scatter(
            results[0, :], results[1, :],
            c=results[2, :], cmap=cmap,
            marker='o', s=8, alpha=0.5,
            label='Random Portfolios'
        )

        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Sharpe Ratio', color=text_color)
        cbar.ax.yaxis.set_tick_params(color=text_color)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=text_color)

        ax.scatter(
            opt_volatility, opt_return,
            marker='*', color=marker_color, s=500, zorder=5,
            label=f'Optimal Portfolio  |  Sharpe: {opt_sharpe:.2f}'
        )

        ax.set_title('Efficient Frontier — Monte Carlo Simulation',
                     color=text_color, pad=14, fontsize=13)
        ax.set_xlabel('Annualized Volatility (Risk)', color=text_color)
        ax.set_ylabel('Annualized Expected Return', color=text_color)
        ax.tick_params(colors=text_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(spine_color)
        ax.grid(True, linestyle='--', alpha=grid_alpha, color=grid_color)
        ax.legend(labelspacing=0.8, facecolor=legend_face,
                  edgecolor=spine_color, labelcolor=text_color)

        plt.tight_layout()

    return fig
