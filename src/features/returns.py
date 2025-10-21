"""Return calculation functions for Index-Coin feature engineering."""

from __future__ import annotations

import pandas as pd


def compute_daily_returns(prices: pd.Series) -> pd.Series:
    """
    Compute daily percentage returns from a price series.

    Parameters:
        prices (pd.Series): Series of prices; must be non-empty, numeric, and contain no negative values.

    Returns:
        pd.Series: Series of daily returns computed as (P_t / P_{t-1}) - 1 with the same index as `prices` (the first element will be `NaN`).

    Raises:
        TypeError: If `prices` is not a pandas Series.
        ValueError: If `prices` is empty, contains non-numeric values, or contains negative values.
    """
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series")
    if len(prices) == 0:
        raise ValueError("prices series cannot be empty")
    if not pd.api.types.is_numeric_dtype(prices):
        raise ValueError("prices must contain numeric values")
    if (prices < 0).any():
        raise ValueError("prices cannot contain negative values")

    # Do not forward/backward fill to avoid silently propagating values.
    return prices.pct_change(fill_method=None)


def compute_cumulative_returns(returns: pd.Series) -> pd.Series:
    """
    Compute cumulative value series from periodic returns.

    Parameters:
        returns (pd.Series): Series of periodic returns (e.g., daily returns). NaN values are allowed and indicate missing returns.

    Returns:
        pd.Series: Cumulative value series with the same index and length as `returns`.
        - If the first return is not NaN, the first element is 1 + first_return
        - If the first return is NaN, the first element is 1.0 and subsequent elements represent the compounded value: V_t = V_{t-1} * (1 + r_{t-1}).
        When a return is NaN, the cumulative value is carried forward unchanged.

    Raises:
        TypeError: If `returns` is not a pandas Series.
        ValueError: If `returns` is empty or contains non-numeric (non-NaN) values.
    """
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")
    if len(returns) == 0:
        raise ValueError("returns series cannot be empty")
    if not pd.api.types.is_numeric_dtype(returns):
        raise ValueError("returns must contain numeric values")

    # Handle the case where first return is NaN (typical from pct_change)
    if pd.isna(returns.iloc[0]):
        # First element is 1.0, then apply returns from second element
        result = (1 + returns.fillna(0)).cumprod()
        result.iloc[0] = 1.0
        return result
    else:
        # First element is 1 + first_return, then compound from there
        return (1 + returns.fillna(0)).cumprod()


def compute_total_return(cumulative_returns: pd.Series) -> float:
    """
    Calculate the total return represented by a cumulative returns series.

    Parameters:
        cumulative_returns (pd.Series): Series of cumulative returns preserving index and length (typically starting at 1.0).

    Returns:
        float: Total return equal to the final cumulative value minus 1.0.
    """
    if not isinstance(cumulative_returns, pd.Series):
        raise TypeError("cumulative_returns must be a pandas Series")
    
    if not pd.api.types.is_numeric_dtype(cumulative_returns):
        raise ValueError("cumulative_returns must contain numeric dtype")

    if len(cumulative_returns) == 0:
        raise ValueError("cumulative_returns series cannot be empty")

    return float(cumulative_returns.iloc[-1] - 1.0)


def annualize_returns(returns: pd.Series, periods_per_year: int = 252) -> pd.Series:
    """
    Convert periodic returns to annualized returns by compounding to the specified number of periods per year.

    Parameters:
        returns (pd.Series): Series of periodic returns (each element is the return for one period). The input index is preserved.
        periods_per_year (int): Number of compounding periods per year (must be greater than 0).

    Returns:
        pd.Series: Series of annualized returns computed as (1 + r) ** periods_per_year - 1 for each input return r, aligned with the input index.

    Raises:
        TypeError: If `returns` is not a pandas Series or contains non-numeric dtype.
        ValueError: If `returns` is empty or `periods_per_year` is not greater than 0.
    """
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")
    if len(returns) == 0:
        raise ValueError("returns series cannot be empty")
    if not pd.api.types.is_numeric_dtype(returns):
        raise ValueError("returns must contain numeric dtype")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    return (1 + returns) ** periods_per_year - 1
