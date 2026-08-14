# 📈 Portfolio Optimizer

A full-stack Python application that leverages **Modern Portfolio Theory (MPT)** to identify the optimal asset allocation for a given set of stocks — maximizing the Sharpe Ratio while supporting custom risk constraints. The application features an interactive **Streamlit** web interface, a Monte Carlo Efficient Frontier visualization, advanced parametric risk metrics, and a historical backtesting engine.

---

## ✨ Features

- **Automated Data Fetching** — Downloads historical Adjusted Close prices for any set of tickers using `yfinance`, with automatic forward/backward fill for missing data.
- **Portfolio Optimization** — Uses `scipy.optimize` (SLSQP method) to find the maximum Sharpe Ratio portfolio under realistic constraints (no short-selling; configurable per-asset weight cap).
- **Efficient Frontier Visualization** — Monte Carlo simulation of 10,000 random portfolios plotted as an interactive scatter chart, with the optimal portfolio highlighted.
- **Advanced Risk Metrics** — Parametric (Gaussian) calculation of **Value at Risk (VaR)** and **Conditional VaR (CVaR / Expected Shortfall)** at a configurable confidence level.
- **Historical Backtesting** — Simulates the growth of a **$10,000** investment over the selected date range, comparing three strategies: Optimized Portfolio, Equal-Weight Portfolio, and a Market Benchmark (e.g., SPY).
- **Interactive Streamlit UI** — Fully configurable sidebar with ticker input, date range, risk-free rate, confidence level, benchmark selection, and a per-asset maximum weight slider.
- **CLI Alternative** — A `main.py` script for running the optimization pipeline directly from the terminal.

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
1. Enter a comma-separated list of stock tickers in the sidebar (e.g., `AAPL, MSFT, GOOGL, AMZN`).
2. Set your desired **Start Date** and **End Date** for the historical data window.
3. Adjust optional parameters: **Risk-Free Rate**, **VaR Confidence Level**, **Benchmark Ticker**, and the **Maximum Weight per Asset** constraint slider.
4. Click **"Run Optimization"**.
5. Review results across two tabs:
   - **📊 Optimization Results** — Optimal weights table, performance metrics, risk metrics, and the Efficient Frontier chart.
   - **📅 Historical Backtest** — Cumulative growth line chart and final portfolio value comparison.

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
    ├── data_fetcher.py     # Fetches Adj. Close prices via yfinance
    ├── metrics_calc.py     # Annualized returns & covariance matrix
    ├── optimizer.py        # Max Sharpe Ratio optimizer (scipy SLSQP)
    ├── visualization.py    # Efficient Frontier plot (Monte Carlo)
    ├── risk_metrics.py     # Parametric VaR & CVaR calculation
    └── backtester.py       # Historical backtesting vs. benchmark
```

---

## 🛠️ Technologies Used

| Library | Purpose |
|---|---|
| **Python 3.x** | Core language |
| **Streamlit** | Interactive web application frontend |
| **yfinance** | Historical market data retrieval |
| **Pandas** | Data manipulation and time-series handling |
| **NumPy** | Numerical computation and Monte Carlo simulation |
| **SciPy** | Constrained portfolio optimization (`scipy.optimize`) |
| **Matplotlib** | Efficient Frontier visualization |

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
