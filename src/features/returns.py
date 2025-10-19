"""
Return calculation functions for Index-Coin feature engineering.
"""

import pandas as pd


def compute_daily_returns(prices: pd.Series) -> pd.Series:
    """
    Compute daily percentage returns from a price series.

    Parameters:
        prices (pd.Series): Series of prices; must be non-empty, numeric, and contain no negative values.

    Returns:
        pd.Series: Series of daily returns computed as (P_t / P_{t-1}) - 1 with the same index as `prices`
        (the first element will be `NaN`).

    Raises:
        TypeError: If `prices` is not a pandas Series.
        ValueError: If `prices` is empty or contains negative values.
    """
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series")
    if len(prices) == 0:
        raise ValueError("prices series cannot be empty")
    if (prices < 0).any():
        raise ValueError("prices cannot contain negative values")

    # No fill method so we don't silently propagate values
    return prices.pct_change(fill_method=None)


def compute_cumulative_returns(returns: pd.Series) -> pd.Series:
    """
    Compute cumulative value series from periodic returns, initializing the starting value to 1.0.

    Parameters:
        returns (pd.Series): Series of periodic returns (e.g., daily returns). NaN values are allowed
            and indicate missing returns.

    Returns:
        pd.Series: Cumulative value series with the same index and length as `returns`.
            The first element is 1.0 and subsequent elements represent the compounded value:
            V_t = V_{t-1} * (1 + r_t). When a return is NaN, the cumulative value is carried forward unchanged.

    Raises:
        TypeError: If `returns` is not a pandas Series.
        ValueError: If `returns` is empty or contains non-numeric (non-NaN) values.
    """
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")
    if len(returns) == 0:
        raise ValueError("returns series cannot be empty")

    numeric_returns = pd.to_numeric(returns, errors="coerce")
    if (numeric_returns.isna() & ~returns.isna()).any():
        raise ValueError("returns must contain only numeric values (NaN allowed)")

    cumulative = pd.Series(index=returns.index, dtype=float)

    # Base value
    cumulative.iloc[0] = 1.0

    # Apply each period's return at its own index
    for i in range(len(returns)):
        if i == 0:
            # For the first period, apply the return to the base value
            if not pd.isna(returns.iloc[i]):
                cumulative.iloc[i] = 1.0 * (1 + float(returns.iloc[i]))
            else:
                cumulative.iloc[i] = 1.0
        else:
            if pd.isna(returns.iloc[i]):
                cumulative.iloc[i] = cumulative.iloc[i - 1]
            else:
                cumulative.iloc[i] = cumulative.iloc[i - 1] * (
                    1 + float(returns.iloc[i])
                )

    return cumulative


def annualize_returns(returns: pd.Series, periods_per_year: int = 252) -> pd.Series:
    """
    Convert periodic returns to annualized returns by compounding to the specified number of periods per year.

    Parameters:
        returns (pd.Series): Series of periodic returns (each element is the return for one period). The input index is preserved.
        periods_per_year (int): Number of compounding periods per year (must be greater than 0).

    Returns:
        pd.Series: Series of annualized returns computed as (1 + r) ** periods_per_year - 1 for each input return r,
            aligned with the input index.

    Raises:
        TypeError: If `returns` is not a pandas Series.
        ValueError: If `periods_per_year` is not greater than 0.
    """
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    return (1 + returns) ** periods_per_year - 1


def compute_total_return(cumulative_returns: pd.Series) -> float:
    """
    Calculate the total return represented by a cumulative returns series.

    Parameters:
        cumulative_returns (pd.Series): Series of cumulative values, typically from compute_cumulative_returns.
            The first value (V_0) should be 1.0, subsequent values track compounded growth.

    Returns:
        float: Total return equal to the final cumulative value minus 1.0.
    """
    if not isinstance(cumulative_returns, pd.Series):
        raise TypeError("cumulative_returns must be a pandas Series")
    if len(cumulative_returns) == 0:
        raise ValueError("cumulative_returns series cannot be empty")

    return float(cumulative_returns.iloc[-1] - 1.0)
