"""Return calculation functions for Index-Coin feature engineering."""

from __future__ import annotations

from typing import List
import pandas as pd

__all__: List[str] = [
    "compute_daily_returns",
    "compute_cumulative_returns",
    "annualize_returns",
    "compute_total_return",
]


def compute_daily_returns(prices: pd.Series) -> pd.Series:
    """Compute period-over-period percentage returns from a price series.

    Args:
        prices (pd.Series): Series of prices; must be non-empty, numeric,
            and contain no negative values.

    Returns:
        pd.Series: Series of returns computed as ``(P_t / P_{t-1}) - 1`` with
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

    # Avoid forward/backward fill to prevent silently propagating values.
    return prices.pct_change(fill_method=None)


def compute_cumulative_returns(returns: pd.Series) -> pd.Series:
    """Convert periodic returns into a cumulative value series starting at 1.0.

    Convention:
        - First value is **1.0** (base NAV).
        - Each subsequent value compounds prior periods’ returns.
        - NaN returns are treated as 0% for compounding (carry forward).

    Args:
        returns (pd.Series): Periodic returns (e.g., daily). NaN allowed.

    Returns:
        pd.Series: Cumulative value series aligned to ``returns``.

    Raises:
        TypeError: If ``returns`` is not a pandas Series.
        ValueError: If ``returns`` is empty or contains non-numeric (non-NaN) values.
    """
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")
    if len(returns) == 0:
        raise ValueError("returns series cannot be empty")

    numeric = pd.to_numeric(returns, errors="coerce")
    # If coercion produced NaN where original wasn't NaN, there were non-numeric values.
    if (numeric.isna() & ~returns.isna()).any():
        raise ValueError("returns must contain only numeric values (NaN allowed)")

    # Treat NaN as 0% return → carry forward; shift so the first element is 1.0.
    cum = (1.0 + numeric.fillna(0.0)).cumprod().shift(fill_value=1.0)
    return cum.astype(float)


def annualize_returns(returns: pd.Series, periods_per_year: int = 252) -> pd.Series:
    """Annualize periodic returns by compounding to the specified frequency.

    Args:
        returns (pd.Series): Periodic returns for one period each.
        periods_per_year (int): Compounding periods per year (> 0).

    Returns:
        pd.Series: Annualized returns computed as ``(1 + r) ** periods_per_year - 1``.

    Raises:
        TypeError: If ``returns`` is not a pandas Series.
        ValueError: If ``periods_per_year`` <= 0.
    """
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    return (1.0 + returns) ** periods_per_year - 1.0


def compute_total_return(cumulative_returns: pd.Series) -> float:
    """Compute total return from a cumulative value series.

    Args:
        cumulative_returns (pd.Series): Cumulative values (V_0 should be 1.0).

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
