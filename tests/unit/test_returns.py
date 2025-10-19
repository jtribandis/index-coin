"""
Unit tests for the returns calculation module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from features.returns import (
    compute_daily_returns,
    compute_cumulative_returns,
    compute_total_return,
)


class TestComputeDailyReturns:
    """Test cases for compute_daily_returns function."""

    def test_compute_daily_returns_basic(self):
        """Test basic daily returns calculation with valid price data."""
        prices = pd.Series(
            [100.0, 102.0, 101.0, 105.0, 108.0],
            index=pd.date_range("2023-01-01", periods=5, freq="D"),
        )
        expected_returns = pd.Series(
            [np.nan, 0.02, -0.009804, 0.039604, 0.02857142857142857], index=prices.index
        )

        result = compute_daily_returns(prices)

        pd.testing.assert_series_equal(
            result, expected_returns, check_exact=False, rtol=1e-5
        )
        assert pd.isna(result.iloc[0])

    def test_compute_daily_returns_with_nans(self):
        """Test daily returns calculation with NaN values in prices."""
        prices = pd.Series(
            [100.0, np.nan, 101.0, 105.0],
            index=pd.date_range("2023-01-01", periods=4, freq="D"),
        )

        result = compute_daily_returns(prices)

        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert pd.isna(result.iloc[2])
        assert not pd.isna(result.iloc[3])

    def test_compute_daily_returns_validation_errors(self):
        """Test input validation for compute_daily_returns."""
        with pytest.raises(TypeError, match="prices must be a pandas Series"):
            compute_daily_returns([100, 102, 101])

        with pytest.raises(ValueError, match="prices series cannot be empty"):
            compute_daily_returns(pd.Series([], dtype=float))


class TestComputeCumulativeReturns:
    """Test cases for compute_cumulative_returns function."""

    def test_compute_cumulative_returns(self):
        """Test cumulative returns calculation."""
        returns = pd.Series(
            [0.02, -0.01, 0.03, -0.005],
            index=pd.date_range("2023-01-01", periods=4, freq="D"),
        )
        expected_cumulative = pd.Series(
            [1.0, 1.02, 1.0098, 1.040094],
            index=pd.date_range("2023-01-01", periods=4, freq="D"),
        )

        result = compute_cumulative_returns(returns)

        pd.testing.assert_series_equal(
            result, expected_cumulative, check_exact=False, rtol=1e-6
        )
        assert result.iloc[0] == 1.0


class TestReturnsRoundtrip:
    """Test roundtrip verification of returns calculations."""

    def test_returns_roundtrip(self):
        """Test that V_T / V_0 equals cumulative return."""
        prices = pd.Series(
            [100.0, 102.0, 101.0, 105.0, 108.0, 110.0],
            index=pd.date_range("2023-01-01", periods=6, freq="D"),
        )

        daily_returns = compute_daily_returns(prices)
        cumulative_returns = compute_cumulative_returns(daily_returns)
        total_return = compute_total_return(cumulative_returns)

        expected_total_return = (prices.iloc[-1] / prices.iloc[0]) - 1

        assert abs(total_return - expected_total_return) < 0.05
