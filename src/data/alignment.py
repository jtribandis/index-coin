"""
Create aligned panel from raw CSVs for Index-Coin backtesting.

This implements Section 2.3 (Calendars & Alignment) from tech spec.
Outputs: data/staging/panel.parquet
"""

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)


def create_business_day_calendar(start_date="2009-01-01", end_date=None):
    """Create business day calendar (NYSE trading days)."""
    if end_date is None:
        end_date = pd.Timestamp.today()

    # Use pandas business day frequency
    calendar = pd.bdate_range(start=start_date, end=end_date, freq="B")

    LOGGER.info(
        f"Created calendar: {len(calendar)} business days "
        f"from {calendar[0].date()} to {calendar[-1].date()}"
    )
    return calendar


def load_raw_csv(symbol, raw_dir="data/raw"):
    """
    Load a single raw CSV from data/raw/.

    Returns DataFrame with single column: {symbol}_adj
    """
    # Handle BTC-USD filename
    filename = f"{symbol}.csv"
    filepath = Path(raw_dir) / filename

    if not filepath.exists():
        LOGGER.warning(f"File not found: {filepath}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(filepath, index_col="Date", parse_dates=True)
    except Exception as e:
        LOGGER.error(f"Failed to load {filepath}: {e}")
        return pd.DataFrame()

    # Use Adj Close (dividend-adjusted)
    if "Adj Close" in df.columns:
        df["adj"] = df["Adj Close"]
    elif "Close" in df.columns:
        df["adj"] = df["Close"]
    else:
        LOGGER.error(f"No price column found for {symbol}")
        return pd.DataFrame()

    # Keep only adj column
    df = df[["adj"]]
    df.columns = [f"{symbol}_adj"]

    LOGGER.info(
        f"Loaded {symbol}: {len(df)} rows from "
        f"{df.index.min().date()} to {df.index.max().date()}"
    )

    return df


def align_to_calendar(df, calendar, asset_name, max_ffill_days=3):
    """
    Align asset data to business day calendar with forward-fill limits.

    Args:
        df: DataFrame with asset price data
        calendar: Business day DatetimeIndex
        asset_name: Asset name (for logging)
        max_ffill_days: Maximum days to forward-fill (3 for ETFs, 1 for BTC)

    Returns:
        DataFrame aligned to calendar
    """
    # Reindex to calendar
    df_aligned = df.reindex(calendar)

    # Count missing before ffill
    missing_before = df_aligned.isna().sum().sum()

    # Forward-fill with limit
    df_aligned = df_aligned.ffill(limit=max_ffill_days)

    # Count missing after ffill
    missing_after = df_aligned.isna().sum().sum()

    if missing_after > 0:
        LOGGER.warning(
            f"{asset_name}: {missing_after} missing values after ffill "
            f"(reduced from {missing_before})"
        )
    else:
        LOGGER.info(
            f"{asset_name}: All {missing_before} gaps filled "
            f"(max ffill: {max_ffill_days} days)"
        )

    return df_aligned


def create_staging_panel(raw_dir="data/raw", output_dir="data/staging"):
    """
    Create aligned panel.parquet from raw CSVs.

    This implements Section 2.3 (Calendars & Alignment) from tech spec.

    Outputs:
        data/staging/panel.parquet - Aligned daily panel
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Step 1: Create business day calendar
    LOGGER.info("=" * 80)
    LOGGER.info("CREATING STAGING PANEL")
    LOGGER.info("=" * 80)
    LOGGER.info("\nStep 1: Creating business day calendar...")

    calendar = create_business_day_calendar(start_date="2009-01-01")

    # Step 2: Load all assets
    symbols = ["GLD", "QQQ", "SPY", "SMH", "IYR", "ANGL", "BTC-USD"]

    LOGGER.info("\nStep 2: Loading raw CSVs...")

    aligned_dfs = []

    for symbol in symbols:
        LOGGER.info(f"\nProcessing {symbol}...")

        # Load raw data
        df = load_raw_csv(symbol, raw_dir=raw_dir)

        if df.empty:
            LOGGER.warning(f"Skipping {symbol} - no data")
            continue

        # Determine ffill limit (DN-5 from tech spec)
        max_ffill = 1 if symbol == "BTC-USD" else 3

        # Align to calendar
        df_aligned = align_to_calendar(
            df, calendar, asset_name=symbol, max_ffill_days=max_ffill
        )

        aligned_dfs.append(df_aligned)

    # Step 3: Concatenate all assets
    LOGGER.info("\nStep 3: Concatenating assets into panel...")
    panel = pd.concat(aligned_dfs, axis=1)

    # Step 4: Save to parquet
    output_path = Path(output_dir) / "panel.parquet"
    panel.to_parquet(output_path)

    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("✓ PANEL CREATED SUCCESSFULLY")
    LOGGER.info("=" * 80)
    LOGGER.info(f"Shape: {panel.shape}")
    LOGGER.info(f"Columns: {list(panel.columns)}")
    LOGGER.info(f"Date range: {panel.index[0].date()} to {panel.index[-1].date()}")
    LOGGER.info(f"Output: {output_path}")
    LOGGER.info("\nMissing values by asset:")
    for col in panel.columns:
        missing = panel[col].isna().sum()
        missing_pct = (missing / len(panel)) * 100
        LOGGER.info(f"  {col}: {missing} ({missing_pct:.2f}%)")

    return panel


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s"
    )

    # Create the staging panel
    panel = create_staging_panel()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Panel shape: {panel.shape}")
    print("\nFirst 5 rows:")
    print(panel.head())
    print("\nLast 5 rows:")
    print(panel.tail())
    print("\n✓ Ready for backtesting!")
