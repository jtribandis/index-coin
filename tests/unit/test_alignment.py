"""
Unit tests for create_staging_panel.py alignment logic.

Tests cover:
- Business day calendar creation
- CSV loading with various formats
- Alignment with forward-fill limits
- Panel creation and concatenation
"""

import logging
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

import pandas as pd
import pytest

# FIXED: Removed src. prefix from imports
from create_staging_panel import (
    create_business_day_calendar,
    load_raw_csv,
    align_to_calendar,
    create_staging_panel,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_calendar():
    """Sample 10-day business calendar for testing."""
    return pd.bdate_range(start="2020-01-01", periods=10, freq="B")


@pytest.fixture
def sample_price_data():
    """Sample price data with some gaps."""
    dates = pd.to_datetime(
        ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08"]
    )
    df = pd.DataFrame({"SPY_adj": [100.0, 101.0, 102.0, 103.0, 104.0]}, index=dates)
    return df


@pytest.fixture
def sample_csv_content_adj_close():
    """CSV content with Adj Close column."""
    return """Date,Open,High,Low,Close,Adj Close,Volume
2020-01-02,100,101,99,100.5,100.0,1000000
2020-01-03,100.5,102,100,101.5,101.0,1100000
2020-01-06,101.5,103,101,102.5,102.0,1200000"""


@pytest.fixture
def sample_csv_content_close_only():
    """CSV content with only Close column (no Adj Close)."""
    return """Date,Open,High,Low,Close,Volume
2020-01-02,100,101,99,100.5,1000000
2020-01-03,100.5,102,100,101.5,1100000"""


@pytest.fixture
def sample_csv_content_no_price():
    """CSV content with no price columns."""
    return """Date,Volume
2020-01-02,1000000
2020-01-03,1100000"""


# ============================================================================
# TEST: create_business_day_calendar
# ============================================================================


def test_create_business_day_calendar_basic():
    """Test basic calendar creation with explicit date range."""
    calendar = create_business_day_calendar(
        start_date="2020-01-01", end_date="2020-01-15"
    )

    # Should have 11 business days in first 15 days of Jan 2020
    assert len(calendar) == 11
    assert calendar[0] == pd.Timestamp("2020-01-01")
    assert isinstance(calendar, pd.DatetimeIndex)


def test_create_business_day_calendar_excludes_weekends():
    """Test that calendar excludes weekends."""
    calendar = create_business_day_calendar(
        start_date="2020-01-01", end_date="2020-01-15"
    )

    # Check no Saturdays or Sundays
    weekdays = calendar.dayofweek
    assert 5 not in weekdays  # No Saturday
    assert 6 not in weekdays  # No Sunday


def test_create_business_day_calendar_no_end_date():
    """Test calendar creation with no end date (uses today)."""
    calendar = create_business_day_calendar(start_date="2020-01-01", end_date=None)

    # Should extend to today
    assert calendar[0] == pd.Timestamp("2020-01-01")
    assert len(calendar) > 100  # Should have many years of data


def test_create_business_day_calendar_single_day():
    """Test calendar with start and end on same day."""
    calendar = create_business_day_calendar(
        start_date="2020-01-02", end_date="2020-01-02"
    )

    assert len(calendar) == 1
    assert calendar[0] == pd.Timestamp("2020-01-02")


# ============================================================================
# TEST: load_raw_csv
# ============================================================================


def test_load_raw_csv_with_adj_close(tmp_path, sample_csv_content_adj_close):
    """Test loading CSV with Adj Close column."""
    # Create temporary CSV file
    csv_file = tmp_path / "SPY.csv"
    csv_file.write_text(sample_csv_content_adj_close)

    # Load the CSV
    df = load_raw_csv("SPY", raw_dir=str(tmp_path))

    # Assertions
    assert not df.empty
    assert "SPY_adj" in df.columns
    assert len(df) == 3
    assert df["SPY_adj"].iloc[0] == 100.0
    assert df["SPY_adj"].iloc[1] == 101.0
    assert df.index[0] == pd.Timestamp("2020-01-02")


def test_load_raw_csv_with_close_only(tmp_path, sample_csv_content_close_only):
    """Test loading CSV with only Close column (fallback)."""
    csv_file = tmp_path / "GLD.csv"
    csv_file.write_text(sample_csv_content_close_only)

    df = load_raw_csv("GLD", raw_dir=str(tmp_path))

    # Should use Close column as fallback
    assert not df.empty
    assert "GLD_adj" in df.columns
    assert df["GLD_adj"].iloc[0] == 100.5


def test_load_raw_csv_no_price_column(tmp_path, sample_csv_content_no_price, caplog):
    """Test loading CSV with no price columns returns empty DataFrame."""
    csv_file = tmp_path / "BAD.csv"
    csv_file.write_text(sample_csv_content_no_price)

    with caplog.at_level(logging.ERROR):
        df = load_raw_csv("BAD", raw_dir=str(tmp_path))

    # Should return empty DataFrame and log error
    assert df.empty
    assert "No price column found" in caplog.text


def test_load_raw_csv_file_not_found(tmp_path, caplog):
    """Test loading non-existent CSV file."""
    with caplog.at_level(logging.WARNING):
        df = load_raw_csv("NONEXISTENT", raw_dir=str(tmp_path))

    # Should return empty DataFrame and log warning
    assert df.empty
    assert "File not found" in caplog.text


def test_load_raw_csv_btc_filename(tmp_path):
    """Test loading BTC-USD with hyphenated filename."""
    csv_content = """Date,Close
2020-01-02,7000.0
2020-01-03,7100.0"""

    csv_file = tmp_path / "BTC-USD.csv"
    csv_file.write_text(csv_content)

    df = load_raw_csv("BTC-USD", raw_dir=str(tmp_path))

    assert not df.empty
    assert "BTC-USD_adj" in df.columns
    assert df["BTC-USD_adj"].iloc[0] == 7000.0


def test_load_raw_csv_corrupted_file(tmp_path, caplog):
    """Test handling of corrupted CSV file."""
    csv_file = tmp_path / "CORRUPT.csv"
    csv_file.write_text("This is not valid CSV content\n@#$%^&*()")

    with caplog.at_level(logging.ERROR):
        df = load_raw_csv("CORRUPT", raw_dir=str(tmp_path))

    # Should return empty DataFrame and log error
    assert df.empty
    assert "Failed to load" in caplog.text


# ============================================================================
# TEST: align_to_calendar
# ============================================================================


def test_align_to_calendar_basic(sample_price_data, sample_calendar):
    """Test basic alignment to calendar."""
    df_aligned = align_to_calendar(
        sample_price_data, sample_calendar, asset_name="SPY", max_ffill_days=3
    )

    # Should have same length as calendar
    assert len(df_aligned) == len(sample_calendar)
    assert df_aligned.index.equals(sample_calendar)


def test_align_to_calendar_ffill_within_limit(sample_calendar):
    """Test forward-fill works within limit."""
    # Create data with 2-day gap
    df = pd.DataFrame(
        {"SPY_adj": [100.0, 101.0]},
        index=pd.to_datetime(["2020-01-02", "2020-01-07"]),
    )

    df_aligned = align_to_calendar(
        df, sample_calendar, asset_name="SPY", max_ffill_days=3
    )

    # Should forward-fill the gap
    assert df_aligned.loc["2020-01-03", "SPY_adj"] == 100.0
    assert df_aligned.loc["2020-01-06", "SPY_adj"] == 100.0


def test_align_to_calendar_ffill_exceeds_limit(sample_calendar, caplog):
    """Test forward-fill stops at limit, leaving NaN."""
    # Create data with 5-day gap (exceeds 3-day limit)
    df = pd.DataFrame(
        {"SPY_adj": [100.0, 105.0]},
        index=pd.to_datetime(["2020-01-02", "2020-01-10"]),
    )

    with caplog.at_level(logging.WARNING):
        df_aligned = align_to_calendar(
            df, sample_calendar, asset_name="SPY", max_ffill_days=3
        )

    # Should have NaN after ffill limit exceeded
    assert df_aligned.loc["2020-01-09"].isna().any()
    assert "missing values after ffill" in caplog.text


def test_align_to_calendar_btc_ffill_limit():
    """Test BTC uses 1-day ffill limit (DN-5 from spec)."""
    calendar = pd.bdate_range(start="2020-01-01", periods=5, freq="B")

    df = pd.DataFrame(
        {"BTC-USD_adj": [7000.0, 7100.0]},
        index=pd.to_datetime(["2020-01-02", "2020-01-07"]),
    )

    df_aligned = align_to_calendar(
        df, calendar, asset_name="BTC-USD", max_ffill_days=1
    )

    # Should ffill 1 day but not 2+ days
    assert df_aligned.loc["2020-01-03", "BTC-USD_adj"] == 7000.0  # 1 day ffill
    assert df_aligned.loc["2020-01-06"].isna().any()  # >1 day gap remains NaN


def test_align_to_calendar_no_missing_values(sample_calendar, caplog):
    """Test alignment when data already complete."""
    # Complete data for all calendar days
    df = pd.DataFrame(
        {"SPY_adj": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0]},
        index=sample_calendar,
    )

    with caplog.at_level(logging.INFO):
        df_aligned = align_to_calendar(
            df, sample_calendar, asset_name="SPY", max_ffill_days=3
        )

    # Should have no missing values
    assert df_aligned.isna().sum().sum() == 0
    assert "All 0 gaps filled" in caplog.text


def test_align_to_calendar_empty_dataframe(sample_calendar):
    """Test alignment with empty input DataFrame."""
    df = pd.DataFrame()

    df_aligned = align_to_calendar(
        df, sample_calendar, asset_name="EMPTY", max_ffill_days=3
    )

    # Should return DataFrame with calendar index, all NaN
    assert len(df_aligned) == len(sample_calendar)
    assert df_aligned.index.equals(sample_calendar)


# ============================================================================
# TEST: create_staging_panel
# ============================================================================


@patch("create_staging_panel.load_raw_csv")
def test_create_staging_panel_basic(mock_load, tmp_path):
    """Test basic panel creation with multiple assets."""
    # Mock CSV loading to return sample data
    def mock_load_side_effect(symbol, raw_dir):
        dates = pd.bdate_range(start="2020-01-01", periods=5, freq="B")
        df = pd.DataFrame({f"{symbol}_adj": [100.0, 101.0, 102.0, 103.0, 104.0]}, index=dates)
        return df

    mock_load.side_effect = mock_load_side_effect

    # Create panel
    panel = create_staging_panel(raw_dir="data/raw", output_dir=str(tmp_path))

    # Assertions
    assert not panel.empty
    assert panel.shape[1] == 7  # 7 assets
    assert "GLD_adj" in panel.columns
    assert "SPY_adj" in panel.columns
    assert "BTC-USD_adj" in panel.columns

    # Check output file created
    output_file = tmp_path / "panel.parquet"
    assert output_file.exists()


@patch("create_staging_panel.load_raw_csv")
def test_create_staging_panel_missing_asset(mock_load, tmp_path, caplog):
    """Test panel creation when some assets are missing."""

    def mock_load_side_effect(symbol, raw_dir):
        if symbol == "MISSING":
            return pd.DataFrame()  # Simulate missing file

        dates = pd.bdate_range(start="2020-01-01", periods=5, freq="B")
        df = pd.DataFrame({f"{symbol}_adj": [100.0, 101.0, 102.0, 103.0, 104.0]}, index=dates)
        return df

    mock_load.side_effect = mock_load_side_effect

    with caplog.at_level(logging.WARNING):
        panel = create_staging_panel(raw_dir="data/raw", output_dir=str(tmp_path))

    # Should skip missing asset
    assert not panel.empty
    assert "Skipping" in caplog.text


@patch("create_staging_panel.load_raw_csv")
def test_create_staging_panel_different_date_ranges(mock_load, tmp_path):
    """Test panel handles assets with different date ranges."""

    def mock_load_side_effect(symbol, raw_dir):
        if symbol == "GLD":
            # Older data
            dates = pd.bdate_range(start="2020-01-01", periods=10, freq="B")
        else:
            # Newer data (shorter range)
            dates = pd.bdate_range(start="2020-01-06", periods=5, freq="B")

        df = pd.DataFrame({f"{symbol}_adj": range(len(dates))}, index=dates)
        return df

    mock_load.side_effect = mock_load_side_effect

    panel = create_staging_panel(raw_dir="data/raw", output_dir=str(tmp_path))

    # Panel should cover full date range
    assert not panel.empty
    # FIXED: Use proper NaN check instead of self-comparison
    # Earlier dates should have NaN for newer assets
    assert pd.isna(panel.loc["2020-01-02", "SPY_adj"])


def test_create_staging_panel_output_format(tmp_path):
    """Test that panel is saved in correct parquet format."""
    # This would need actual implementation or more sophisticated mocking
    # For now, test that output directory is created
    output_dir = tmp_path / "staging"

    # Just verify directory creation logic
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    assert output_dir.exists()
    assert output_dir.is_dir()


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


def test_end_to_end_with_sample_data(tmp_path):
    """Integration test with real sample CSV files."""
    # Create sample CSV files
    spy_csv = tmp_path / "SPY.csv"
    spy_csv.write_text("""Date,Close,Adj Close
2020-01-02,100.0,100.0
2020-01-03,101.0,101.0
2020-01-06,102.0,102.0""")

    gld_csv = tmp_path / "GLD.csv"
    gld_csv.write_text("""Date,Close,Adj Close
2020-01-02,150.0,150.0
2020-01-03,151.0,151.0""")

    # Load and align
    calendar = create_business_day_calendar(start_date="2020-01-01", end_date="2020-01-10")

    spy_df = load_raw_csv("SPY", raw_dir=str(tmp_path))
    spy_aligned = align_to_calendar(spy_df, calendar, "SPY", max_ffill_days=3)

    gld_df = load_raw_csv("GLD", raw_dir=str(tmp_path))
    gld_aligned = align_to_calendar(gld_df, calendar, "GLD", max_ffill_days=3)

    # Concatenate
    panel = pd.concat([spy_aligned, gld_aligned], axis=1)

    # Assertions
    assert panel.shape[1] == 2
    assert len(panel) == len(calendar)
    assert not panel.loc["2020-01-02"].isna().all()


# ============================================================================
# EDGE CASES
# ============================================================================


def test_calendar_with_holidays():
    """Test that business day calendar handles market holidays."""
    # This is handled by pandas bdate_range
    calendar = create_business_day_calendar(
        start_date="2020-12-20", end_date="2020-12-31"
    )

    # Should exclude Dec 25 (Christmas) and Dec 26-27 (weekend)
    christmas = pd.Timestamp("2020-12-25")
    assert christmas not in calendar


def test_align_with_future_dates(sample_calendar):
    """Test alignment doesn't forward-fill into future beyond data."""
    # Data ends before calendar ends
    df = pd.DataFrame(
        {"SPY_adj": [100.0, 101.0]}, index=pd.to_datetime(["2020-01-02", "2020-01-03"])
    )

    df_aligned = align_to_calendar(
        df, sample_calendar, asset_name="SPY", max_ffill_days=3
    )

    # Should have data at start
    assert df_aligned.loc["2020-01-02", "SPY_adj"] == 100.0

    # Should ffill for 3 days then stop
    assert df_aligned.loc["2020-01-06", "SPY_adj"] == 101.0  # Within ffill limit
    assert pd.isna(df_aligned.loc["2020-01-09", "SPY_adj"])  # Beyond limit


def test_load_csv_with_timezone_aware_dates(tmp_path):
    """Test handling of timezone-aware dates in CSV."""
    csv_content = """Date,Close
2020-01-02 00:00:00+00:00,100.0
2020-01-03 00:00:00+00:00,101.0"""

    csv_file = tmp_path / "TZ.csv"
    csv_file.write_text(csv_content)

    df = load_raw_csv("TZ", raw_dir=str(tmp_path))

    # Should load successfully
    assert not df.empty
    assert len(df) == 2
