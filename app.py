import datetime as dt
import streamlit as st
import pandas as pd

from src.data_fetcher import get_stock_data
from src.metrics_calc import calculate_annual_returns, calculate_covariance_matrix
from src.optimizer import optimize_portfolio
from src.visualization import plot_efficient_frontier
from src.risk_metrics import calculate_parametric_var, calculate_parametric_cvar
from src.backtester import run_backtest

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Portfolio Optimizer", layout="wide", page_icon="📈")

# ── AVAILABLE TICKERS ──────────────────────────────────────────────────────────
AVAILABLE_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM',
    'V', 'WMT', 'JNJ', 'PG', 'MA', 'UNH', 'HD', 'BAC', 'XOM', 'KO',
    'PEP', 'ABBV', 'MRK', 'LLY', 'AVGO', 'COST', 'ORCL', 'NFLX',
    'ADBE', 'CSCO', 'CRM', 'AMD',
]

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <h1 style='margin-bottom:0'>📈 Portfolio Optimizer</h1>
    <p style='color:#888888; margin-top:4px; font-size:15px'>
        Modern Portfolio Theory · Max Sharpe Ratio · Efficient Frontier · Risk Metrics · Backtesting
    </p>
    <hr style='border-color:#222222; margin-bottom:24px'>
    """,
    unsafe_allow_html=True,
)

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## ⚙️ Configuration")

tickers = st.sidebar.multiselect(
    "Select Stocks to Simulate",
    options=AVAILABLE_TICKERS,
    default=['AAPL', 'MSFT', 'GOOGL', 'AMZN']
)

today = dt.datetime.today().date()
default_start = today - dt.timedelta(days=5 * 365)

st.sidebar.markdown("**Date Range**")
col_s, col_e = st.sidebar.columns(2)
start_date = col_s.date_input("Start", value=default_start)
end_date = col_e.date_input("End", value=today)

st.sidebar.markdown("**Parameters**")
risk_free_rate = st.sidebar.number_input("Risk-Free Rate", value=0.02, step=0.005, format="%.3f")
confidence_level = st.sidebar.number_input(
    "VaR/CVaR Confidence Level", value=0.95, step=0.01, max_value=0.99, min_value=0.50
)
benchmark_ticker = st.sidebar.text_input("Benchmark Ticker", value="SPY")
max_weight = st.sidebar.slider(
    "Max Weight per Asset",
    min_value=0.10, max_value=1.0, value=1.0, step=0.05, format="%.2f"
)
user_capital = st.sidebar.number_input(
    "Initial Capital (USD)", min_value=100.0, value=10000.0, step=1000.0, format="%.2f"
)

st.sidebar.markdown("---")
run_btn = st.sidebar.button("🚀 Run Optimization", type="primary", use_container_width=True)

# ── MAIN EXECUTION ─────────────────────────────────────────────────────────────
if run_btn:
    if len(tickers) < 2:
        st.error("Please select at least **two** stocks.")
    elif max_weight * len(tickers) < 1.0:
        st.error(
            f"Invalid constraint: **{max_weight:.0%}** cap × **{len(tickers)} assets** = "
            f"**{max_weight * len(tickers):.0%}** max allocation — cannot reach 100%. "
            "Increase the weight cap or add more tickers."
        )
        st.stop()
    else:
        with st.spinner("Fetching data and running optimization..."):
            try:
                # Pipeline
                df_prices = get_stock_data(
                    tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
                )
                mean_returns = calculate_annual_returns(df_prices)
                cov_matrix = calculate_covariance_matrix(df_prices)
                results = optimize_portfolio(
                    mean_returns, cov_matrix,
                    risk_free_rate=risk_free_rate,
                    max_weight=max_weight
                )
                var_val = calculate_parametric_var(
                    results['expected_return'], results['volatility'],
                    confidence_level=confidence_level
                )
                cvar_val = calculate_parametric_cvar(
                    results['expected_return'], results['volatility'],
                    confidence_level=confidence_level
                )

            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.stop()

        # ── KPI ROW ────────────────────────────────────────────────────────────
        st.markdown("### Key Performance Indicators")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Expected Annual Return",  f"{results['expected_return'] * 100:.2f}%")
        k2.metric("Annual Volatility",       f"{results['volatility'] * 100:.2f}%")
        k3.metric("Max Sharpe Ratio",        f"{results['max_sharpe_ratio']:.4f}")
        k4.metric(f"VaR ({int(confidence_level*100)}%)",  f"{var_val * 100:.2f}%")
        k5.metric(f"CVaR ({int(confidence_level*100)}%)", f"{cvar_val * 100:.2f}%")

        st.markdown("---")

        # ── TABS ───────────────────────────────────────────────────────────────
        tab1, tab2 = st.tabs(["📊 Optimization Results", "📅 Historical Backtest"])

        # ─ TAB 1 ──────────────────────────────────────────────────────────────
        with tab1:
            left, right = st.columns([1, 2], gap="large")

            with left:
                st.markdown("#### Optimal Asset Allocation")
                weights_df = (results['optimal_weights'] * 100).rename("Weight (%)")
                st.bar_chart(weights_df, color="#a3ff00")

                st.markdown("#### Weights Table")
                display_df = weights_df.to_frame()
                display_df["Weight (%)"] = display_df["Weight (%)"].map("{:.2f}%".format)
                st.dataframe(display_df, use_container_width=True)

            with right:
                st.markdown("#### Efficient Frontier")
                fig = plot_efficient_frontier(
                    mean_returns, cov_matrix, results['optimal_weights'],
                    risk_free_rate=risk_free_rate
                )
                st.pyplot(fig, transparent=False)

        # ─ TAB 2 ──────────────────────────────────────────────────────────────
        with tab2:
            st.markdown(
                f"Growth of a **$10,000** investment · Optimized vs. Equal-Weight vs. **{benchmark_ticker.upper()}**"
            )
            try:
                backtest_df = run_backtest(
                    df_prices, results['optimal_weights'],
                    benchmark_ticker=benchmark_ticker.strip().upper(),
                    initial_capital=user_capital
                )
                st.line_chart(backtest_df, color=["#a3ff00", "#00bfff", "#ff6b6b"])

                st.markdown("#### Final Portfolio Values")
                final = backtest_df.iloc[-1]
                b1, b2, b3 = st.columns(3)
                b1.metric(
                    "📈 Optimized Portfolio",
                    f"${final['Optimized Portfolio']:,.2f}",
                    delta=f"${final['Optimized Portfolio'] - 10_000:,.2f}"
                )
                b2.metric(
                    "⚖️ Equal Weight",
                    f"${final['Equal Weight']:,.2f}",
                    delta=f"${final['Equal Weight'] - 10_000:,.2f}"
                )
                bench_col = f"Benchmark ({benchmark_ticker.strip().upper()})"
                b3.metric(
                    f"🏦 {bench_col}",
                    f"${final[bench_col]:,.2f}",
                    delta=f"${final[bench_col] - 10_000:,.2f}"
                )
            except ValueError as e:
                st.error(f"Backtest failed: {e}")
            except Exception as e:
                st.error(f"Unexpected backtest error: {e}")

else:
    # ── LANDING PLACEHOLDER ────────────────────────────────────────────────────
    st.markdown(
        """
        <div style='text-align:center; padding: 80px 0; color:#444444'>
            <p style='font-size:56px'>📊</p>
            <p style='font-size:20px'>Configure your portfolio in the sidebar and click <b style='color:#a3ff00'>Run Optimization</b> to begin.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
