"""Index-Coin data ingestion utilities.

Provides functions to fetch historical data from Yahoo Finance using yfinance,
persist raw CSVs under `data/raw/`, and generate a manifest of ingested files.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

try:
    import yfinance as yf
except ImportError as import_error:
    yf = None
    logging.getLogger(__name__).warning("yfinance import failed: %s", import_error)


LOGGER = logging.getLogger(__name__)
RAW_DIR = os.path.join("data", "raw")


def _ensure_raw_dir() -> None:
    """Ensure the raw data directory exists."""
    os.makedirs(RAW_DIR, exist_ok=True)


def _normalize_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance columns to required schema."""
    if df is None or df.empty:
        return pd.DataFrame()

    normalized = df.copy()
    try:
        normalized.index = pd.to_datetime(normalized.index)
    except (ValueError, TypeError):
        # Only ignore errors related to invalid datetime conversion.
        pass
    normalized.index.name = "Date"

    if "Stock Splits" in normalized.columns and "Splits" not in normalized.columns:
        normalized = normalized.rename(columns={"Stock Splits": "Splits"})

    expected = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "Dividends",
        "Splits",
    ]
    for col in expected:
        if col not in normalized.columns:
            if col in ("Dividends", "Splits"):
                normalized[col] = 0.0
            else:
                normalized[col] = pd.NA

    normalized = normalized[expected]
    return normalized


def fetch_yahoo_data(
    symbol: str, start_date: str, end_date: Optional[str] = None
) -> pd.DataFrame:
    """Fetch historical data for a symbol from Yahoo Finance."""
    if yf is None:
        LOGGER.error("yfinance is not available; cannot fetch %s", symbol)
        return pd.DataFrame()

    try:
        LOGGER.info(
            "Fetching %s from Yahoo Finance: %s -> %s",
            symbol,
            start_date,
            end_date or "today",
        )
        df = yf.download(
            tickers=symbol,
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=False,
            actions=True,
            progress=False,
            threads=True,
        )

        if isinstance(df, pd.DataFrame) and not df.empty:
            df = _normalize_yf_columns(df)
        else:
            LOGGER.error("No data returned for symbol %s", symbol)
            return pd.DataFrame()

        try:
            if end_date is not None:
                mask = (df.index >= pd.to_datetime(start_date)) & (
                    df.index <= pd.to_datetime(end_date)
                )
            else:
                mask = df.index >= pd.to_datetime(start_date)
            df = df.loc[mask]
        except (ValueError, TypeError, KeyError) as date_error:
            # Only catch errors related to invalid dates or indexing operations.
            LOGGER.warning("Date filtering failed for %s: %s", symbol, date_error)

        return df
    except Exception as exc:
        LOGGER.exception("Failed to fetch %s: %s", symbol, exc)
        return pd.DataFrame()


def fetch_all_symbols(start_date: str = "2009-01-01") -> Dict[str, Dict[str, object]]:
    """Fetch a predefined set of symbols and save each to CSV under data/raw/."""
    _ensure_raw_dir()

    symbols = ["GLD", "QQQ", "SPY", "SMH", "IYR", "ANGL", "BTC-USD"]
    summary: Dict[str, Dict[str, object]] = {}

    for symbol in symbols:
        try:
            LOGGER.info("Processing symbol %s", symbol)
            df = fetch_yahoo_data(symbol=symbol, start_date=start_date, end_date=None)
            if df.empty:
                LOGGER.warning("Skipping %s due to empty dataset", symbol)
                summary[symbol] = {"rows": 0, "start": None, "end": None}
                continue

            csv_path = os.path.join(RAW_DIR, f"{symbol}.csv")
            try:
                df.to_csv(csv_path, index=True)
                LOGGER.info("Saved %s rows to %s", len(df), csv_path)
            except Exception as save_error:
                LOGGER.exception("Failed to save %s: %s", csv_path, save_error)

            date_min = df.index.min()
            date_max = df.index.max()
            summary[symbol] = {
                "rows": int(len(df)),
                "start": None
                if pd.isna(date_min)
                else pd.to_datetime(date_min).date().isoformat(),
                "end": None
                if pd.isna(date_max)
                else pd.to_datetime(date_max).date().isoformat(),
            }
        except Exception as exc:
            LOGGER.exception("Error while processing %s: %s", symbol, exc)
            summary[symbol] = {"rows": 0, "start": None, "end": None}

    return summary


def fetch_btc_data(start_date: str = "2010-07-17") -> pd.DataFrame:
    """Fetch BTC-USD daily data using yfinance and save to `data/raw/BTC.csv`."""
    _ensure_raw_dir()
    df = fetch_yahoo_data("BTC-USD", start_date=start_date, end_date=None)
    if df.empty:
        LOGGER.error("BTC-USD returned empty data")
        return df

    csv_path = os.path.join(RAW_DIR, "BTC.csv")
    try:
        df.to_csv(csv_path, index=True)
        LOGGER.info("Saved BTC data (%s rows) to %s", len(df), csv_path)
    except Exception as save_error:
        LOGGER.exception("Failed to save BTC data: %s", save_error)

    return df


def _sha256_file(path: str) -> str:
    """Compute SHA-256 hash of a file's contents."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_manifest() -> Dict[str, Dict[str, object]]:
    """Generate a manifest JSON summarizing CSV files under `data/raw/`."""
    _ensure_raw_dir()

    manifest: Dict[str, Dict[str, object]] = {}
    try:
        for name in sorted(os.listdir(RAW_DIR)):
            if not name.lower().endswith(".csv"):
                continue
            path = os.path.join(RAW_DIR, name)
            if not os.path.isfile(path):
                continue

            try:
                df = pd.read_csv(path, parse_dates=["Date"])
            except (OSError, pd.errors.ParserError, ValueError) as read_error:
                # Only catch file I/O and CSV parsing related errors; let others bubble up.
                LOGGER.warning("Skipping unreadable CSV %s: %s", path, read_error)
                continue

            rows = int(len(df))
            date_min_val: Optional[datetime] = None
            date_max_val: Optional[datetime] = None
            if rows > 0 and "Date" in df.columns:
                try:
                    date_min_val = pd.to_datetime(df["Date"].min()).to_pydatetime()
                    date_max_val = pd.to_datetime(df["Date"].max()).to_pydatetime()
                except Exception as date_parse_error:
                    LOGGER.warning(
                        "Date parsing failed for %s: %s", path, date_parse_error
                    )

            try:
                sha256 = _sha256_file(path)
            except Exception as hash_error:
                LOGGER.warning("Hashing failed for %s: %s", path, hash_error)
                sha256 = ""

            try:
                size_kb = os.path.getsize(path) / 1024.0
            except Exception as size_error:
                LOGGER.warning("Size retrieval failed for %s: %s", path, size_error)
                size_kb = 0.0

            symbol = os.path.splitext(os.path.basename(path))[0]
            manifest[symbol] = {
                "rows": rows,
                "date_min": None
                if date_min_val is None
                else date_min_val.date().isoformat(),
                "date_max": None
                if date_max_val is None
                else date_max_val.date().isoformat(),
                "sha256": sha256,
                "size_kb": round(size_kb, 1),
            }
    except FileNotFoundError:
        LOGGER.warning("Raw data directory not found: %s", RAW_DIR)

    manifest_path = os.path.join(RAW_DIR, "_ingest_manifest.json")
    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        LOGGER.info(
            "Wrote manifest with %s entries to %s", len(manifest), manifest_path
        )
    except Exception as write_error:
        LOGGER.exception("Failed to write manifest: %s", write_error)

    return manifest


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    summary = fetch_all_symbols()
    print(f"Fetched {len(summary)} symbols")
    manifest = generate_manifest()
    print(f"Manifest created with {len(manifest)} entries")
