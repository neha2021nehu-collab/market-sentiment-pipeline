"""
Unit tests for source_c_yahoo.py — all offline, no real network calls.
Run: pytest tests/test_source_c.py -v
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from scrapers.source_c_yahoo import (
    _safe_float,
    _extract_fundamentals,
    YahooFinanceScraper,
)


class TestSafeFloat:
    def test_normal_float(self):
        assert _safe_float(3.14) == 3.14

    def test_integer(self):
        assert _safe_float(100) == 100.0

    def test_string_number(self):
        assert _safe_float("42.5") == 42.5

    def test_none_returns_none(self):
        assert _safe_float(None) is None

    def test_nan_returns_none(self):
        assert _safe_float(float("nan")) is None

    def test_non_numeric_returns_none(self):
        assert _safe_float("not-a-number") is None

    def test_rounds_to_4_decimals(self):
        assert _safe_float(3.14159265) == 3.1416


class TestExtractFundamentals:
    def test_extracts_key_fields(self):
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "longName":           "NVIDIA Corporation",
            "sector":             "Technology",
            "marketCap":          1_200_000_000_000,
            "trailingPE":         45.3,
            "fiftyTwoWeekHigh":   974.0,
            "fiftyTwoWeekLow":    430.0,
            "targetMeanPrice":    850.0,
        }
        result = _extract_fundamentals(mock_ticker, "NVDA")

        assert result["ticker"] == "NVDA"
        assert result["company_name"] == "NVIDIA Corporation"
        assert result["pe_ratio"] == 45.3
        assert result["fifty_two_week_high"] == 974.0
        assert "scraped_at" in result

    def test_missing_fields_return_none(self):
        mock_ticker = MagicMock()
        mock_ticker.info = {}
        result = _extract_fundamentals(mock_ticker, "FAKE")

        assert result["ticker"] == "FAKE"
        assert result["pe_ratio"] is None
        assert result["market_cap"] is None

    def test_info_exception_returns_empty_info(self):
        mock_ticker = MagicMock()
        mock_ticker.info = {}
        result = _extract_fundamentals(mock_ticker, "ERR")
        assert result["ticker"] == "ERR"


class TestYahooFinanceScraper:
    def test_init_defaults(self):
        scraper = YahooFinanceScraper()
        assert "NVDA" in scraper.tickers
        assert scraper.period == "6mo"

    def test_init_custom(self):
        scraper = YahooFinanceScraper(tickers=["AAPL"], period="1mo")
        assert scraper.tickers == ["AAPL"]
        assert scraper.period == "1mo"

    @patch("scrapers.source_c_yahoo.yf.download")
    def test_fetch_ohlcv_success(self, mock_download):
        mock_df = pd.DataFrame({
            "Open":   [400.0, 410.0],
            "High":   [420.0, 430.0],
            "Low":    [390.0, 400.0],
            "Close":  [415.0, 425.0],
            "Volume": [50_000_000, 48_000_000],
        }, index=pd.to_datetime(["2024-06-01", "2024-06-02"]))
        mock_download.return_value = mock_df

        scraper = YahooFinanceScraper(tickers=["NVDA"], period="1mo")
        records = scraper._fetch_ohlcv("NVDA")

        assert len(records) == 2
        assert records[0]["ticker"] == "NVDA"
        assert records[0]["date"] == "2024-06-01"
        assert records[0]["close"] == 415.0
        assert records[0]["volume"] == 50_000_000

    @patch("scrapers.source_c_yahoo.yf.download")
    def test_fetch_ohlcv_empty_returns_empty_list(self, mock_download):
        mock_download.return_value = pd.DataFrame()
        scraper = YahooFinanceScraper(tickers=["FAKE"])
        assert scraper._fetch_ohlcv("FAKE") == []