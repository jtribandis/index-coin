"""Unit tests for src.data.ingest module."""

import os

import pandas as pd
import pytest
from unittest.mock import patch, mock_open

from src.data.ingest import (
    fetch_yahoo_data,
    fetch_all_symbols,
    fetch_btc_data,
    generate_manifest,
)


@pytest.fixture
def sample_yahoo_data():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "Open": [100.0] * 10,
            "High": [101.0] * 10,
            "Low": [99.0] * 10,
            "Close": [100.5] * 10,
            "Adj Close": [100.5] * 10,
            "Volume": [1000000] * 10,
            "Dividends": [0.0] * 10,
            "Stock Splits": [0.0] * 10,
        },
        index=dates,
    )


@pytest.fixture
def temp_raw_dir(tmp_path, monkeypatch):
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    monkeypatch.setattr("src.data.ingest.RAW_DIR", str(raw_dir))
    return raw_dir


@patch("src.data.ingest.yf.download")
def test_fetch_yahoo_data_success(mock_download, sample_yahoo_data):
    mock_download.return_value = sample_yahoo_data.copy()
    df = fetch_yahoo_data("SPY", "2020-01-01", "2020-01-31")

    expected_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "Dividends",
        "Splits",
    ]
    assert not df.empty
    assert list(df.columns) == expected_cols
    assert isinstance(df.index, pd.DatetimeIndex)


@patch("src.data.ingest.yf.download")
def test_fetch_yahoo_data_invalid_symbol(mock_download):
    mock_download.return_value = None
    df_none = fetch_yahoo_data("INVALID", "2020-01-01", "2020-01-31")
    assert isinstance(df_none, pd.DataFrame)
    assert df_none.empty


@patch("src.data.ingest.os.makedirs")
@patch("src.data.ingest.open", new_callable=mock_open)
@patch("pandas.DataFrame.to_csv")
@patch("src.data.ingest.fetch_yahoo_data")
def test_fetch_all_symbols(
    mock_fetch_yahoo_data,
    mock_to_csv,
    mock_file_open,
    mock_makedirs,
    temp_raw_dir,
    sample_yahoo_data,
):
    def df_for_symbol(symbol):
        df = sample_yahoo_data.copy()
        df = df.rename(columns={"Stock Splits": "Splits"})
        return df

    symbols = ["GLD", "QQQ", "SPY", "SMH", "IYR", "ANGL", "BTC-USD"]
    df_map = {s: df_for_symbol(s) for s in symbols}

    def side_effect(symbol, start_date, end_date=None):
        return df_map.get(symbol, pd.DataFrame())

    mock_fetch_yahoo_data.side_effect = side_effect
    summary = fetch_all_symbols("2020-01-01")

    for s in symbols:
        assert s in summary
        entry = summary[s]
        assert "rows" in entry
        assert "start" in entry
        assert "end" in entry


@patch("src.data.ingest.os.makedirs")
@patch("pandas.DataFrame.to_csv")
@patch("src.data.ingest.fetch_yahoo_data")
def test_fetch_btc_data(
    mock_fetch_yahoo_data, mock_to_csv, mock_makedirs, temp_raw_dir, sample_yahoo_data
):
    df = sample_yahoo_data.copy().rename(columns={"Stock Splits": "Splits"})
    mock_fetch_yahoo_data.return_value = df
    result_df = fetch_btc_data("2020-01-01")
    assert result_df.equals(df)


def _write_sample_csv(path, dates):
    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": [1.0] * len(dates),
            "Close": [1.5] * len(dates),
        }
    )
    df.to_csv(path, index=False)


def test_generate_manifest(temp_raw_dir, tmp_path):
    dates1 = pd.date_range("2020-01-01", periods=5, freq="D")
    file1 = os.path.join(str(temp_raw_dir), "SPY.csv")
    _write_sample_csv(file1, dates1)

    manifest = generate_manifest()

    assert isinstance(manifest, dict)
    assert "SPY" in manifest
    assert manifest["SPY"]["rows"] == 5
    assert len(manifest["SPY"]["sha256"]) == 64


def test_generate_manifest_empty_directory(temp_raw_dir):
    for filename in os.listdir(str(temp_raw_dir)):
        os.remove(os.path.join(str(temp_raw_dir), filename))

    manifest = generate_manifest()
    assert isinstance(manifest, dict)
    assert len(manifest) == 0
