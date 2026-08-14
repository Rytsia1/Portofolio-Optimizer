import numpy as np
import pandas as pd
import yfinance as yf


def run_backtest(
    prices_df: pd.DataFrame,
    optimal_weights: pd.Series,
    benchmark_ticker: str = "SPY",
    initial_capital: float = 10_000.0
) -> pd.DataFrame:
    """
    Simulates the historical growth of $10,000 across three scenarios:
    Optimized Portfolio, Equal-Weight Portfolio, and a Market Benchmark.

    Args:
        prices_df (pd.DataFrame): Adjusted Close prices from data_fetcher.
        optimal_weights (pd.Series): Optimal asset weights from the optimizer.
        benchmark_ticker (str): Benchmark ticker symbol. Defaults to "SPY".
        initial_capital (float): Initial investment amount in USD. Defaults to 10,000.

    Returns:
        pd.DataFrame: DataFrame with cumulative portfolio values for all three scenarios,
                      indexed by Date, with columns: "Optimized Portfolio",
                      "Equal Weight", and "Benchmark (SPY)".

    Raises:
        ValueError: If the benchmark data cannot be downloaded or is empty.
    """
    # Determine the date range from prices_df
    start_date = prices_df.index.min()
    end_date = prices_df.index.max()

    # --- Download Benchmark Data ---
    benchmark_raw = yf.download(
        benchmark_ticker,
        start=start_date,
        end=end_date,
        progress=False
    )

    if benchmark_raw.empty:
        raise ValueError(
            f"Could not download benchmark data for '{benchmark_ticker}'. "
            "Please check your internet connection or try a different ticker."
        )

    # Extract Adj Close (fall back to Close)
    bench_col = "Adj Close" if "Adj Close" in benchmark_raw.columns else "Close"
    benchmark_prices = benchmark_raw[bench_col].squeeze()
    benchmark_prices = benchmark_prices.ffill().bfill()

    # --- Daily Returns ---
    portfolio_returns = prices_df.pct_change().dropna()

    # Align benchmark to the same dates as portfolio_returns
    benchmark_returns = benchmark_prices.pct_change().reindex(portfolio_returns.index).dropna()
    portfolio_returns = portfolio_returns.reindex(benchmark_returns.index)

    # --- Weights ---
    num_assets = len(prices_df.columns)
    equal_weights = np.array([1.0 / num_assets] * num_assets)
    opt_weights_array = optimal_weights.reindex(prices_df.columns).fillna(0.0).values

    # --- Cumulative Portfolio Values ---
    optimized_daily = portfolio_returns.dot(opt_weights_array)
    equal_daily = portfolio_returns.dot(equal_weights)

    cumulative_optimized = initial_capital * (1 + optimized_daily).cumprod()
    cumulative_equal = initial_capital * (1 + equal_daily).cumprod()
    cumulative_benchmark = initial_capital * (1 + benchmark_returns).cumprod()

    # --- Combine into a single DataFrame ---
    result_df = pd.DataFrame({
        "Optimized Portfolio": cumulative_optimized,
        "Equal Weight": cumulative_equal,
        f"Benchmark ({benchmark_ticker})": cumulative_benchmark,
    }).dropna()

    return result_df
