"""
Returns calculation module for Index-Coin feature engineering.
"""

import pandas as pd


def compute_daily_returns(prices: pd.Series) -> pd.Series:
    """
    Compute daily returns from price series.

    Formula: r_t = (P_t / P_{t-1}) - 1
    """
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series")

    if len(prices) == 0:
        raise ValueError("prices series cannot be empty")

    if not pd.api.types.is_numeric_dtype(prices):
        raise ValueError("prices must contain numeric values")

    if (prices < 0).any():
        raise ValueError("prices cannot contain negative values")

    return prices.pct_change(fill_method=None)


def compute_cumulative_returns(returns: pd.Series) -> pd.Series:
    """
    Compute cumulative returns from daily returns series.

    Formula: V_t = V_{t-1} * (1 + r_t) with V_0 = 1

    Returns a series with the SAME length and index as the input.
    """
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")

    if len(returns) == 0:
        raise ValueError("returns series cannot be empty")

    numeric_returns = pd.to_numeric(returns, errors="coerce")
    if numeric_returns.isna().any() and not returns.isna().any():
        raise ValueError("returns must contain only numeric values (NaN allowed)")

    # Create cumulative series with SAME index as input
    cumulative = pd.Series(index=returns.index, dtype=float)

    # First value is V_0 = 1.0
    cumulative.iloc[0] = 1.0

    # Calculate cumulative returns: V_t = V_{t-1} * (1 + r_{t-1})
    for i in range(1, len(returns)):
        if pd.isna(returns.iloc[i - 1]):
            cumulative.iloc[i] = cumulative.iloc[i - 1]
        else:
            cumulative.iloc[i] = cumulative.iloc[i - 1] * (
                1 + float(returns.iloc[i - 1])
            )

    return cumulative


def annualize_returns(returns: pd.Series, periods_per_year: int = 252) -> pd.Series:
    """Annualize returns by scaling to yearly frequency."""
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    return (1 + returns) ** periods_per_year - 1


def compute_total_return(cumulative_returns: pd.Series) -> float:
    """Compute total return from cumulative returns series."""
    if not isinstance(cumulative_returns, pd.Series):
        raise TypeError("cumulative_returns must be a pandas Series")

    if len(cumulative_returns) == 0:
        raise ValueError("cumulative_returns series cannot be empty")

    return float(cumulative_returns.iloc[-1] - 1.0)
