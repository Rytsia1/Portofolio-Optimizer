import datetime as dt
import streamlit as st
import pandas as pd

from src.data_fetcher import get_stock_data
from src.metrics_calc import calculate_annual_returns, calculate_covariance_matrix
from src.optimizer import optimize_portfolio
from src.visualization import plot_efficient_frontier
from src.risk_metrics import calculate_parametric_var, calculate_parametric_cvar
from src.backtester import run_backtest

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Portfolio Optimizer", layout="wide", page_icon="📈")

st.title("📈 Portfolio Optimizer & Risk Dashboard")
st.markdown("A robust financial engineering pipeline using Modern Portfolio Theory (Markowitz) and Parametric Risk Bounds.")

# --- SIDEBAR ---
st.sidebar.header("Configuration")

tickers_input = st.sidebar.text_input(
    "Stock Tickers (comma-separated)",
    value="AAPL, MSFT, GOOGL, AMZN"
)

today = dt.datetime.today().date()
default_start = today - dt.timedelta(days=5 * 365)

start_date = st.sidebar.date_input("Start Date", value=default_start)
end_date = st.sidebar.date_input("End Date", value=today)

risk_free_rate = st.sidebar.number_input("Risk-Free Rate", value=0.02, step=0.005, format="%.3f")
confidence_level = st.sidebar.number_input(
    "VaR/CVaR Confidence Level", value=0.95, step=0.01, max_value=0.99, min_value=0.50
)
benchmark_ticker = st.sidebar.text_input("Benchmark Ticker", value="SPY")
max_weight = st.sidebar.slider(
    "Maximum Weight per Asset (Constraint)",
    min_value=0.10, max_value=1.0, value=1.0, step=0.05,
    format="%.2f"
)

# --- MAIN EXECUTION ---
if st.sidebar.button("Run Optimization", type="primary"):
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

    if len(tickers) < 2:
        st.error("Please enter at least two stock tickers to optimize a portfolio.")
    elif max_weight * len(tickers) < 1.0:
        st.error(
            f"Invalid constraint: a maximum weight of **{max_weight:.0%}** across "
            f"**{len(tickers)} assets** can only cover **{max_weight * len(tickers):.0%}** "
            "of the portfolio — less than 100%. Please increase the maximum weight or add more tickers."
        )
        st.stop()
    else:
        with st.spinner("Fetching market data and running optimizations..."):
            try:
                # 1. Fetch Data
                df_prices = get_stock_data(
                    tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
                )

                # 2. Calculate Metrics
                mean_returns = calculate_annual_returns(df_prices)
                cov_matrix = calculate_covariance_matrix(df_prices)

                # 3. Optimize Portfolio
                results = optimize_portfolio(mean_returns, cov_matrix, risk_free_rate=risk_free_rate, max_weight=max_weight)

                # 4. Advanced Risk Metrics
                var_95 = calculate_parametric_var(
                    results['expected_return'], results['volatility'], confidence_level=confidence_level
                )
                cvar_95 = calculate_parametric_cvar(
                    results['expected_return'], results['volatility'], confidence_level=confidence_level
                )

                st.success("Optimization Complete!")

                # --- TABS ---
                tab1, tab2 = st.tabs(["📊 Optimization Results", "📅 Historical Backtest"])

                # ── TAB 1: Optimization Results ──────────────────────────────
                with tab1:
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.subheader("Optimal Asset Allocation")
                        weights_df = results['optimal_weights'].to_frame("Weight")
                        weights_df['Weight (%)'] = (weights_df['Weight'] * 100).map("{:.2f}%".format)
                        st.dataframe(weights_df[['Weight (%)']], use_container_width=True)

                        st.subheader("Performance Metrics")
                        st.metric("Expected Annual Return", f"{results['expected_return'] * 100:.2f}%")
                        st.metric("Annual Volatility (Risk)", f"{results['volatility'] * 100:.2f}%")
                        st.metric("Max Sharpe Ratio", f"{results['max_sharpe_ratio']:.4f}")

                        st.subheader(f"Risk Metrics ({int(confidence_level * 100)}% Confidence)")
                        st.metric("Value at Risk (VaR)", f"{var_95 * 100:.2f}%")
                        st.metric("Conditional VaR (CVaR)", f"{cvar_95 * 100:.2f}%")

                    with col2:
                        st.subheader("Efficient Frontier")
                        fig = plot_efficient_frontier(
                            mean_returns,
                            cov_matrix,
                            results['optimal_weights'],
                            risk_free_rate=risk_free_rate
                        )
                        st.pyplot(fig)

                # ── TAB 2: Historical Backtest ────────────────────────────────
                with tab2:
                    st.subheader("Historical Backtest: $10,000 Growth Simulation")
                    st.markdown(
                        f"Compares your **Optimized Portfolio** vs. an **Equal-Weight Portfolio** "
                        f"vs. the **{benchmark_ticker} Benchmark** over the selected date range."
                    )

                    try:
                        backtest_df = run_backtest(
                            df_prices,
                            results['optimal_weights'],
                            benchmark_ticker=benchmark_ticker.strip().upper()
                        )

                        # Line chart
                        st.line_chart(backtest_df, use_container_width=True)

                        # Final values summary
                        st.subheader("Final Portfolio Values")
                        final_values = backtest_df.iloc[-1]
                        bcol1, bcol2, bcol3 = st.columns(3)

                        bcol1.metric(
                            "📈 Optimized Portfolio",
                            f"${final_values['Optimized Portfolio']:,.2f}",
                            delta=f"${final_values['Optimized Portfolio'] - 10_000:,.2f}"
                        )
                        bcol2.metric(
                            "⚖️ Equal Weight",
                            f"${final_values['Equal Weight']:,.2f}",
                            delta=f"${final_values['Equal Weight'] - 10_000:,.2f}"
                        )
                        bench_col_name = f"Benchmark ({benchmark_ticker.strip().upper()})"
                        bcol3.metric(
                            f"🏦 {bench_col_name}",
                            f"${final_values[bench_col_name]:,.2f}",
                            delta=f"${final_values[bench_col_name] - 10_000:,.2f}"
                        )

                    except ValueError as e:
                        st.error(f"Backtest failed: {str(e)}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred during backtesting: {str(e)}")

            except Exception as e:
                st.error(f"An error occurred during execution: {str(e)}")
else:
    st.info("👈 Configure your portfolio settings in the sidebar and click 'Run Optimization' to begin calculating.")
