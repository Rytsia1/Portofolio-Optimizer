import datetime as dt
import streamlit as st
import pandas as pd

from src.data_fetcher import get_stock_data
from src.metrics_calc import calculate_annual_returns, calculate_covariance_matrix
from src.optimizer import optimize_portfolio
from src.visualization import plot_efficient_frontier
from src.risk_metrics import calculate_parametric_var, calculate_parametric_cvar

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
default_start = today - dt.timedelta(days=5*365)

start_date = st.sidebar.date_input("Start Date", value=default_start)
end_date = st.sidebar.date_input("End Date", value=today)

risk_free_rate = st.sidebar.number_input("Risk-Free Rate", value=0.02, step=0.005, format="%.3f")
confidence_level = st.sidebar.number_input("VaR/CVaR Confidence Level", value=0.95, step=0.01, max_value=0.99, min_value=0.50)

# --- MAIN EXECUTION ---
if st.sidebar.button("Run Optimization", type="primary"):
    # Parse tickers
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(',') if ticker.strip()]
    
    if len(tickers) < 2:
        st.error("Please enter at least two stock tickers to optimize a portfolio.")
    else:
        with st.spinner("Fetching market data and running optimizations..."):
            try:
                # 1. Fetch Data
                df_prices = get_stock_data(tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                
                # 2. Calculate Metrics
                mean_returns = calculate_annual_returns(df_prices)
                cov_matrix = calculate_covariance_matrix(df_prices)
                
                # 3. Optimize Portfolio
                results = optimize_portfolio(mean_returns, cov_matrix, risk_free_rate=risk_free_rate)
                
                # 4. Advanced Risk Metrics
                var_95 = calculate_parametric_var(results['expected_return'], results['volatility'], confidence_level=confidence_level)
                cvar_95 = calculate_parametric_cvar(results['expected_return'], results['volatility'], confidence_level=confidence_level)
                
                st.success("Optimization Complete!")
                
                # --- DASHBOARD RENDER ---
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("Optimal Asset Allocation")
                    # Format dataframe to percentage nicely
                    weights_df = results['optimal_weights'].to_frame("Weight")
                    weights_df['Weight (%)'] = (weights_df['Weight'] * 100).map("{:.2f}%".format)
                    st.dataframe(weights_df[['Weight (%)']], use_container_width=True)
                    
                    st.subheader("Performance Metrics")
                    st.metric("Expected Annual Return", f"{results['expected_return'] * 100:.2f}%")
                    st.metric("Annual Volatility (Risk)", f"{results['volatility'] * 100:.2f}%")
                    st.metric("Max Sharpe Ratio", f"{results['max_sharpe_ratio']:.4f}")
                    
                    st.subheader(f"Risk Metrics ({int(confidence_level*100)}% Confidence)")
                    st.metric("Value at Risk (VaR)", f"{var_95 * 100:.2f}%")
                    st.metric("Conditional VaR (CVaR)", f"{cvar_95 * 100:.2f}%")
                    
                with col2:
                    st.subheader("Efficient Frontier")
                    # Render the matplotlib figure passed directly from src/visualization.py
                    fig = plot_efficient_frontier(
                        mean_returns, 
                        cov_matrix, 
                        results['optimal_weights'],
                        risk_free_rate=risk_free_rate
                    )
                    st.pyplot(fig)
                    
            except Exception as e:
                st.error(f"An error occurred during execution: {str(e)}")
else:
    st.info("👈 Configure your portfolio settings in the sidebar and click 'Run Optimization' to begin calculating.")
