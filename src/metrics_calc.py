import pandas as pd

def calculate_annual_returns(df: pd.DataFrame) -> pd.Series:
    """
    Calculates the annualized historical mean returns for each asset.
    Assumes 252 trading days in a year.

    Args:
        df (pd.DataFrame): DataFrame of asset prices.

    Returns:
        pd.Series: Annualized mean returns for each asset.
    """
    # Calculate daily percentage changes and drop the first NaN row
    daily_returns = df.pct_change().dropna()
    
    # Calculate annualized mean returns
    annual_returns = daily_returns.mean() * 252
    
    return annual_returns

def calculate_covariance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the annualized covariance matrix between the assets.
    Assumes 252 trading days in a year.

    Args:
        df (pd.DataFrame): DataFrame of asset prices.

    Returns:
        pd.DataFrame: Annualized covariance matrix.
    """
    # Calculate daily percentage changes and drop the first NaN row
    daily_returns = df.pct_change().dropna()
    
    # Calculate annualized covariance matrix
    annual_covariance = daily_returns.cov() * 252
    
    return annual_covariance
