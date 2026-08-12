import datetime as dt
from src.data_fetcher import get_stock_data
from src.metrics_calc import calculate_annual_returns, calculate_covariance_matrix
from src.optimizer import optimize_portfolio

def main():
    # 1. Define sample tickers and date range (5 years ago to today)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    
    today = dt.datetime.today()
    five_years_ago = today - dt.timedelta(days=5 * 365)
    
    end_date = today.strftime('%Y-%m-%d')
    start_date = five_years_ago.strftime('%Y-%m-%d')
    
    print(f"Fetching historical data for: {', '.join(tickers)}")
    print(f"Date Horizon: {start_date} to {end_date}...\n")
    
    # 2. Fetch the stock data
    df_prices = get_stock_data(tickers, start_date, end_date)
    print("Data successfully fetched.\n")
    
    # 3. Calculate historical metrics
    mean_returns = calculate_annual_returns(df_prices)
    cov_matrix = calculate_covariance_matrix(df_prices)
    
    # 4. Run portfolio optimization
    print("Running optimization algorithm...\n")
    results = optimize_portfolio(mean_returns, cov_matrix)
    
    # 5. Print results in a clean, human-readable format
    print("=" * 45)
    print("        PORTFOLIO OPTIMIZATION RESULTS       ")
    print("=" * 45)
    
    print("\n[ Optimal Asset Allocation ]")
    # Iterate over the pandas Series and format as percentages
    for ticker, weight in results['optimal_weights'].items():
        print(f"  {ticker}: {weight * 100:.2f}%")
        
    print("\n[ Expected Portfolio Performance ]")
    print(f"  Expected Annual Return : {results['expected_return'] * 100:.2f}%")
    print(f"  Annual Volatility      : {results['volatility'] * 100:.2f}%")
    print(f"  Max Sharpe Ratio       : {results['max_sharpe_ratio']:.4f}")
    
    print("=" * 45)

if __name__ == "__main__":
    main()
