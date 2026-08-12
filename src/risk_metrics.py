import scipy.stats as stats

def calculate_parametric_var(expected_return: float, volatility: float, confidence_level: float = 0.95) -> float:
    """
    Calculates the parametric Value at Risk (VaR) of a portfolio as a positive percentage loss.

    Args:
        expected_return (float): Expected portfolio return (annualized).
        volatility (float): Expected portfolio volatility (annualized).
        confidence_level (float, optional): The confidence level for the VaR. Defaults to 0.95.

    Returns:
        float: The VaR threshold as a positive float representing maximum expected loss percentage.
    """
    # Inverse normal CDF to find the standard deviation multiplier (z-score) at the tail
    z_score = stats.norm.ppf(1 - confidence_level)
    
    var_threshold = expected_return + (z_score * volatility)
    
    # Return as a positive loss representation
    return -var_threshold if var_threshold < 0 else 0.0

def calculate_parametric_cvar(expected_return: float, volatility: float, confidence_level: float = 0.95) -> float:
    """
    Calculates the Conditional Value at Risk (CVaR) / Expected Shortfall of a portfolio using the parametric approach.

    Args:
        expected_return (float): Expected portfolio return (annualized).
        volatility (float): Expected portfolio volatility (annualized).
        confidence_level (float, optional): The confidence level for the CVaR. Defaults to 0.95.

    Returns:
        float: The CVaR as a positive float representing the expected average loss beyond the VaR threshold.
    """
    z_score = stats.norm.ppf(1 - confidence_level)
    
    # PDF density evaluation exactly at the quantile cut-off
    pdf_eval = stats.norm.pdf(z_score)
    
    # Parametric formula for Expected Shortfall given a Gaussian distribution
    cvar_threshold = expected_return - (volatility * (pdf_eval / (1 - confidence_level)))
    
    return -cvar_threshold if cvar_threshold < 0 else 0.0
