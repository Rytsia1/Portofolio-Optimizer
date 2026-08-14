import numpy as np
import pandas as pd
import plotly.graph_objects as go
from src.optimizer import portfolio_performance

def plot_efficient_frontier(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    optimal_weights: pd.Series,
    risk_free_rate: float = 0.02,
    num_portfolios: int = 10000
) -> go.Figure:
    """
    Plots the Efficient Frontier using Monte Carlo simulation via Plotly.
    The chart background is left transparent/unset so it seamlessly 
    inherits the native Streamlit Light/Dark theme.

    Args:
        mean_returns (pd.Series): Annualized mean returns for each asset.
        cov_matrix (pd.DataFrame): Annualized covariance matrix.
        optimal_weights (pd.Series): Optimal portfolio weights.
        risk_free_rate (float): Risk-free rate. Defaults to 0.02.
        num_portfolios (int): Number of random portfolios. Defaults to 10000.

    Returns:
        go.Figure: Plotly figure object.
    """
    num_assets = len(mean_returns)
    results = np.zeros((3, num_portfolios))

    # Run Monte Carlo
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

    fig = go.Figure()

    # Trace 1: Random Portfolios (Scatter)
    fig.add_trace(go.Scatter(
        x=results[0, :],
        y=results[1, :],
        mode='markers',
        marker=dict(
            color=results[2, :],
            colorscale='Blues',
            showscale=True,
            size=5,
            opacity=0.6,
            colorbar=dict(title="Sharpe Ratio")
        ),
        name='Random Portfolios',
        hoverinfo='x+y+text',
        text=[f"Sharpe: {s:.2f}" for s in results[2, :]]
    ))

    # Trace 2: Optimal Portfolio (Electric Blue Star)
    fig.add_trace(go.Scatter(
        x=[opt_volatility],
        y=[opt_return],
        mode='markers',
        marker=dict(
            symbol='star',
            color='#3b82f6',  # Electric Blue accent
            size=22,
            line=dict(color='black', width=1)
        ),
        name=f'Optimal Portfolio (Max Sharpe: {opt_sharpe:.2f})',
        hovertemplate=(
            f"<b>Max Sharpe: {opt_sharpe:.2f}</b><br>"
            "Volatility: %{x:.2%}<br>"
            "Expected Return: %{y:.2%}<extra></extra>"
        )
    ))

    # Layout configuration — background left alone for native theme sync
    fig.update_layout(
        title="Efficient Frontier — Monte Carlo Simulation",
        xaxis_title="Annualized Volatility (Risk)",
        yaxis_title="Annualized Expected Return",
        xaxis=dict(tickformat=".2%"),
        yaxis=dict(tickformat=".2%"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="closest",
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig
