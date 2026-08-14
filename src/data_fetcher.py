import pandas as pd
import yfinance as yf
from typing import List

def get_stock_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads historical 'Adj Close' price data for a list of tickers.

    Args:
        tickers (List[str]): List of stock ticker symbols.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: Cleaned DataFrame of adjusted close prices with missing 
                      values handled (forward-fill then backward-fill).
    """
    # Download data from Yahoo Finance
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    
    # Extract 'Adj Close' prices
    price_col = 'Adj Close' if 'Adj Close' in data else 'Close'
    df = data[price_col]
        
    # Standardize to DataFrame if a single ticker was provided
    if isinstance(df, pd.Series):
        df = df.to_frame(name=tickers[0])
        
    # Handle missing values: forward fill first, then backward fill
    clean_df = df.ffill().bfill()
    
    # Drop tickers that failed to download entirely
    clean_df = clean_df.dropna(axis=1, how='all')
    
    if clean_df.empty or clean_df.shape[1] < 2:
        raise ValueError(
            "Failed to fetch sufficient data from Yahoo Finance. "
            "This is usually due to a Rate Limit (Too Many Requests), or invalid tickers."
        )
    
    return clean_df
