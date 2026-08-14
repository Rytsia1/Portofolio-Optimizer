# 📈 Portfolio Optimizer

A full-stack Python application that leverages **Modern Portfolio Theory (MPT)** to identify the optimal asset allocation for a given set of stocks — maximizing the Sharpe Ratio while supporting custom risk constraints. The application features an interactive **Streamlit** web interface, a Monte Carlo Efficient Frontier visualization via Plotly, advanced parametric risk metrics, and a dedicated historical Stress Testing module.

---

## ✨ Features

- **Automated Data Fetching** — Downloads historical Adjusted Close prices for any set of tickers using `yfinance`, with automatic forward/backward fill for missing data. Implements native Streamlit caching to bypass rate limits and prevent redundant API calls.
- **Portfolio Optimization** — Uses `scipy.optimize` (SLSQP method) to find the maximum Sharpe Ratio portfolio under realistic constraints (no short-selling; configurable per-asset weight cap).
- **Interactive Visualization** — Monte Carlo simulation of 10,000 random portfolios plotted as a dynamic, interactive Plotly scatter chart, natively inheriting Streamlit's Light/Dark mode themes.
- **Advanced Risk Metrics** — Parametric (Gaussian) calculation of **Value at Risk (VaR)** and **Conditional VaR (CVaR / Expected Shortfall)** at a configurable confidence level.
- **Historical Backtesting** — Simulates the growth of a custom investment amount across the selected date range (extending back to Jan 2000), comparing the Optimized Portfolio, Equal-Weight Portfolio, and a Market Benchmark (e.g., SPY).
- **Stress Testing Module** — Simulates portfolio resilience across predefined historical market crashes (e.g., COVID-19 Crash, 2022 Tech Bear Market, 2008 Financial Crisis) and outputs comparative Maximum Drawdown (MDD) metrics.
- **Production-Ready UI** — Configurable sidebar, tabbed results view, session-state persistence, responsive KPI rows, and direct Streamlit Cloud deployment readiness.

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/portfolio-optimizer.git
cd portfolio-optimizer
```

### 2. Create and Activate a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🖥️ Usage

### Streamlit Web App (Recommended)

```bash
streamlit run streamlit_app.py
```

Open your browser and navigate to **http://localhost:8501**.

**How to use the UI:**
1. Enter your desired stock tickers in the sidebar multiselect box.
2. Set your desired **Start Date** and **End Date** for the historical data window.
3. Adjust optional parameters: **Risk-Free Rate**, **VaR Confidence Level**, **Benchmark Ticker**, **Initial Capital**, and the **Maximum Weight per Asset** constraint slider.
4. Click **"Run Optimization"**.
5. Review results across three tabs:
   - **📊 Optimization Results** — Optimal weights table, performance metrics, risk metrics, and the interactive Plotly Efficient Frontier chart.
   - **📅 Historical Backtest** — Cumulative growth line chart and final portfolio value comparisons.
   - **⚠️ Stress Testing** — Select historical crises to simulate and compare maximum drawdowns against the market.

### CLI Alternative

```bash
python main.py
```

Outputs the optimal weights, expected return, volatility, Sharpe Ratio, VaR, and CVaR directly to the terminal.

---

## 📁 Project Structure

```
portfolio-optimizer/
│
├── streamlit_app.py        # Streamlit web application (main UI)
├── main.py                 # CLI execution script
├── requirements.txt        # Python dependencies
├── .gitignore
│
└── src/
    ├── __init__.py         # Package root
    ├── data_fetcher.py     # Cached yfinance Adj Close price retrieval
    ├── metrics_calc.py     # Annualized returns & covariance matrix
    ├── optimizer.py        # Max Sharpe Ratio optimizer (scipy SLSQP)
    ├── visualization.py    # Plotly Efficient Frontier & Stress charts
    ├── risk_metrics.py     # Parametric VaR & CVaR calculation
    └── backtester.py       # Backtesting and Crisis Scenario MDD calculations
```

---

## 🛠️ Technologies Used

| Library | Purpose |
|---|---|
| **Python 3.x** | Core language |
| **Streamlit** | Interactive web application frontend & Session State control |
| **Plotly** | Interactive visualization and theming |
| **yfinance** | Historical market data retrieval |
| **Pandas** | Data manipulation and time-series handling |
| **NumPy** | Numerical computation and Monte Carlo simulation |
| **SciPy** | Constrained portfolio optimization (`scipy.optimize`) |

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
