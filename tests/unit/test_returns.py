"""Unit tests for return calculation functions."""

import numpy as np
import pandas as pd
import pytest

from src.features.returns import (
    annualize_returns,
    compute_cumulative_returns,
    compute_daily_returns,
    compute_total_return,
)


class TestComputeDailyReturns:
    """Tests for compute_daily_returns function."""

    def test_compute_daily_returns_basic(self):
        """Test basic daily returns calculation."""
        prices = pd.Series(
            [100.0, 102.0, 101.0, 105.0],
            index=pd.date_range("2023-01-01", periods=4, freq="D"),
        )
        expected_returns = pd.Series(
            [np.nan, 0.02, -0.00980392, 0.03960396],
            index=pd.date_range("2023-01-01", periods=4, freq="D"),
        )

        result = compute_daily_returns(prices)

        pd.testing.assert_series_equal(
            result, expected_returns, check_exact=False, rtol=1e-6
        )
        assert pd.isna(result.iloc[0])

    def test_compute_daily_returns_validation_errors(self):
        """Test input validation for compute_daily_returns."""
        with pytest.raises(TypeError, match="prices must be a pandas Series"):
            compute_daily_returns([100, 102, 101])

        with pytest.raises(ValueError, match="prices series cannot be empty"):
            compute_daily_returns(pd.Series([], dtype=float))

    def test_compute_daily_returns_negative_prices(self):
        """Test that negative prices raise ValueError."""
        prices = pd.Series([100.0, -50.0, 101.0])
        with pytest.raises(ValueError, match="prices cannot contain negative values"):
            compute_daily_returns(prices)


class TestComputeCumulativeReturns:
    """Tests for compute_cumulative_returns function."""

    def test_compute_cumulative_returns(self):
        """Test cumulative returns calculation."""
        returns = pd.Series(
            [0.02, -0.01, 0.03, -0.005],
            index=pd.date_range("2023-01-01", periods=4, freq="D"),
        )
        expected_cumulative = pd.Series(
            [1.02, 1.0098, 1.040094, 1.034893],
            index=pd.date_range("2023-01-01", periods=4, freq="D"),
        )

        result = compute_cumulative_returns(returns)

        pd.testing.assert_series_equal(
            result, expected_cumulative, check_exact=False, rtol=1e-6
        )
        assert abs(result.iloc[0] - 1.02) < 1e-9

    def test_compute_cumulative_returns_with_leading_nan(self):
        """Test cumulative returns when first return is NaN (typical from pct_change)."""
        returns = pd.Series(
            [np.nan, 0.02, -0.01, 0.03],
            index=pd.date_range("2023-01-01", periods=4, freq="D"),
        )

        result = compute_cumulative_returns(returns)

        assert result.iloc[0] == 1.0
        assert abs(result.iloc[1] - 1.02) < 1e-9
        assert abs(result.iloc[2] - 1.0098) < 1e-9
        assert abs(result.iloc[3] - 1.040094) < 1e-6


class TestReturnsRoundtrip:
    """Tests for roundtrip consistency."""

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

        assert abs(total_return - expected_total_return) < 1e-9, (
            f"Roundtrip mismatch: total_return={total_return:.10f}, "
            f"expected={expected_total_return:.10f}"
        )


class TestAnnualizeReturns:
    """Tests for annualize_returns function."""

    def test_annualize_returns_basic(self):
        """Test basic annualization with default periods_per_year=252."""
        returns = pd.Series([0.01, 0.02, -0.005])
        result = annualize_returns(returns)

        expected = pd.Series(
            [
                (1 + 0.01) ** 252 - 1,
                (1 + 0.02) ** 252 - 1,
                (1 + -0.005) ** 252 - 1,
            ]
        )

        pd.testing.assert_series_equal(result, expected, check_exact=False, rtol=1e-6)

    def test_annualize_returns_custom_periods(self):
        """Test annualization with custom periods_per_year."""
        returns = pd.Series([0.01, 0.02])
        result = annualize_returns(returns, periods_per_year=12)

        expected = pd.Series(
            [
                (1 + 0.01) ** 12 - 1,
                (1 + 0.02) ** 12 - 1,
            ]
        )

        pd.testing.assert_series_equal(result, expected, check_exact=False, rtol=1e-6)

    def test_annualize_returns_zero_returns(self):
        """Test annualization with zero returns."""
        returns = pd.Series([0.0, 0.0, 0.0])
        result = annualize_returns(returns)

        expected = pd.Series([0.0, 0.0, 0.0])

        pd.testing.assert_series_equal(result, expected)

    def test_annualize_returns_negative_returns(self):
        """Test annualization with negative returns."""
        returns = pd.Series([-0.01, -0.02])
        result = annualize_returns(returns)

        expected = pd.Series(
            [
                (1 + -0.01) ** 252 - 1,
                (1 + -0.02) ** 252 - 1,
            ]
        )

        pd.testing.assert_series_equal(result, expected, check_exact=False, rtol=1e-6)

    def test_annualize_returns_with_nan(self):
        """Test annualization handles NaN values."""
        returns = pd.Series([0.01, np.nan, 0.02])
        result = annualize_returns(returns)

        assert pd.isna(result.iloc[1])
        assert abs(result.iloc[0] - ((1 + 0.01) ** 252 - 1)) < 1e-9
        assert abs(result.iloc[2] - ((1 + 0.02) ** 252 - 1)) < 1e-9

    def test_annualize_returns_validation_errors(self):
        """Test input validation for annualize_returns."""
        with pytest.raises(TypeError, match="returns must be a pandas Series"):
            annualize_returns([0.01, 0.02])

        with pytest.raises(ValueError, match="periods_per_year must be positive"):
            annualize_returns(pd.Series([0.01]), periods_per_year=0)

        with pytest.raises(ValueError, match="periods_per_year must be positive"):
            annualize_returns(pd.Series([0.01]), periods_per_year=-5)
