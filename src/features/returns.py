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
    """Convert periodic returns into a cumulative value series starting at 1.0.

    The cumulative value series V is defined as:
        V_t = Π_{i=0..t} (1 + r_i), treating NaN returns as 0 for compounding
        (i.e., the cumulative value is carried forward unchanged on NaN).

    Note:
        With this convention, the first element equals ``1.0 * (1 + r_0)`` if
        ``r_0`` is not NaN; otherwise it remains ``1.0``.

    Args:
        returns (pd.Series): Series of periodic returns (e.g., daily returns). NaN values
            are allowed and indicate missing returns (carry-forward behavior).

    Returns:
        pd.Series: Cumulative value series with the same index as ``returns``.

    Raises:
        TypeError: If ``returns`` is not a pandas Series.
        ValueError: If ``returns`` is empty or contains non-numeric (non-NaN) values.
    """
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")
    if len(returns) == 0:
        raise ValueError("returns series cannot be empty")

    numeric = pd.to_numeric(returns, errors="coerce")
    # If coercion produced NaN where original wasn't NaN, then non-numeric existed.
    if (numeric.isna() & ~returns.isna()).any():
        raise ValueError("returns must contain only numeric values (NaN allowed)")

    # Treat NaN as a 0% return → carry forward the cumulative value.
    return (1.0 + numeric.fillna(0.0)).cumprod()


def annualize_returns(returns: pd.Series, periods_per_year: int = 252) -> pd.Series:
    """Annualize periodic returns by compounding to the specified frequency.

    Args:
        returns (pd.Series): Series of periodic returns (each element is the return for one period).
        periods_per_year (int): Number of compounding periods per year (must be > 0).

    Returns:
        pd.Series: Annualized returns computed as ``(1 + r) ** periods_per_year - 1``.

    Raises:
        TypeError: If ``returns`` is not a pandas Series.
        ValueError: If ``periods_per_year`` is not greater than 0.
    """
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    return (1.0 + returns) ** periods_per_year - 1.0


def compute_total_return(cumulative_returns: pd.Series) -> float:
    """Compute total return from a cumulative value series.

    Args:
        cumulative_returns (pd.Series): Series of cumulative values, typically produced by
            ``compute_cumulative_returns``. The first value (V_0) should be 1.0.

    Returns:
        float: Total return equal to the final cumulative value minus 1.0.

    Raises:
        TypeError: If ``cumulative_returns`` is not a pandas Series.
        ValueError: If ``cumulative_returns`` is empty.
    """
    if not isinstance(cumulative_returns, pd.Series):
        raise TypeError("cumulative_returns must be a pandas Series")
    if len(cumulative_returns) == 0:
        raise ValueError("cumulative_returns series cannot be empty")

    return float(cumulative_returns.iloc[-1] - 1.0)
