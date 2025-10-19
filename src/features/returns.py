"""Return calculation functions for Index-Coin feature engineering."""

from __future__ import annotations

import pandas as pd


def compute_daily_returns(prices: pd.Series) -> pd.Series:
    """Compute period-over-period percentage returns from a price series.

    Args:
        prices (pd.Series): Series of prices; must be non-empty, numeric, and contain no negative values.

    Returns:
        pd.Series: Series of daily returns computed as ``(P_t / P_{t-1}) - 1`` with
            the same index as ``prices``. The first element will be ``NaN``.

    Raises:
        TypeError: If ``prices`` is not a pandas Series.
        ValueError: If ``prices`` is empty or contains negative values.
    """
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series")
    if len(prices) == 0:
        raise ValueError("prices series cannot be empty")
    if (prices < 0).any():
        raise ValueError("prices cannot contain negative values")

    # Do not forward/backward fill to avoid silently propagating values.
    return prices.pct_change(fill_method=None)


def compute_cumulative_returns(returns: pd.Series) -> pd.Series:
    """Compute cumulative returns from a series of period returns.

    Args:
        returns (pd.Series): Series of period returns. NaN values are treated as zero returns.

    Returns:
        pd.Series: Series of cumulative returns computed as the cumulative product of (1 + r_t),
            with the same index as ``returns``.
    """
    return (1 + returns.fillna(0)).cumprod()


def compute_total_return(cumulative_returns: pd.Series) -> float:
    """Compute total return from cumulative returns series.

    Args:
        cumulative_returns (pd.Series): Series of cumulative returns.

    Returns:
        float: Total return computed as the final cumulative return minus 1.
    """
    final_value: float = cumulative_returns.iloc[-1]
    return final_value - 1


def annualize_returns(returns: pd.Series, periods_per_year: int = 252) -> pd.Series:
    """Annualize period returns using compound growth.

    Args:
        returns (pd.Series): Series of period returns to annualize.
        periods_per_year (int): Number of periods per year. Defaults to 252 (trading days).

    Returns:
        pd.Series: Series of annualized returns computed as ``(1 + r)^periods_per_year - 1``
            with the same index as ``returns``.

    Raises:
        TypeError: If ``returns`` is not a pandas Series.
        ValueError: If ``periods_per_year`` is not positive.
    """
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    return (1 + returns) ** periods_per_year - 1
