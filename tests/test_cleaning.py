"""
Unit tests for analysis/cleaning.py — all offline, uses mock DataFrames.
Run: pytest tests/test_cleaning.py -v
"""

import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, mock_open
from analysis.cleaning import (
    clean_source_a,
    clean_source_b,
    clean_source_c,
    SENTIMENT_SCORE_MAP,
)

# ---------------------------------------------------------------------------
# Fixtures — minimal valid raw data matching real scraper output
# ---------------------------------------------------------------------------

SOURCE_A_DATA = [
    {
        "headline": "NVDA surges on AI demand  ",
        "summary": "Nvidia reports strong earnings",
        "url": "https://example.com/nvda",
        "published_at": "2024-06-01T10:00:00+00:00",
        "tickers_mentioned": ["NVDA"],
        "source": "techcrunch_ai",
        "scraped_at": "2024-06-01T10:05:00+00:00",
    },
    {
        "headline": "MSFT and GOOGL compete in AI",
        "summary": "Cloud battle heats up",
        "url": "https://example.com/msft-googl",
        "published_at": "2024-06-02T08:00:00+00:00",
        "tickers_mentioned": ["MSFT", "GOOGL"],
        "source": "ars_technica",
        "scraped_at": "2024-06-02T08:05:00+00:00",
    },
    # Duplicate — should be dropped
    {
        "headline": "NVDA surges on AI demand",
        "summary": "Nvidia reports strong earnings",
        "url": "https://example.com/nvda",
        "published_at": "2024-06-01T10:00:00+00:00",
        "tickers_mentioned": ["NVDA"],
        "source": "techcrunch_ai",
        "scraped_at": "2024-06-01T10:05:00+00:00",
    },
]

SOURCE_B_DATA = [
    {
        "title": "NVDA beats expectations",
        "summary": "Strong quarter for Nvidia",
        "url": "https://example.com/nvda-earnings",
        "source": "TechCrunch",
        "published_at": "20240601T100000",
        "ticker": "NVDA",
        "overall_sentiment_label": "bullish",
        "overall_sentiment_score": 0.35,
        "ticker_sentiment_label": "bullish",
        "ticker_sentiment_score": 0.42,
        "data_source": "alphavantage_news_sentiment",
        "scraped_at": "2024-06-01T10:05:00+00:00",
    },
    {
        "title": "MSFT faces headwinds",
        "summary": "Cloud growth slows",
        "url": "https://example.com/msft-headwinds",
        "source": "Reuters",
        "published_at": "20240602T090000",
        "ticker": "MSFT",
        "overall_sentiment_label": "bearish",
        "overall_sentiment_score": -0.2,
        "ticker_sentiment_label": "bearish",
        "ticker_sentiment_score": -0.3,
        "data_source": "alphavantage_news_sentiment",
        "scraped_at": "2024-06-02T09:05:00+00:00",
    },
]

SOURCE_C_OHLCV = [
    {"ticker": "NVDA", "date": "2024-06-01", "open": 800.0, "high": 820.0,
     "low": 795.0, "close": 815.0, "volume": 40000000, "scraped_at": "2024-06-01T10:00:00+00:00"},
    {"ticker": "NVDA", "date": "2024-06-02", "open": 815.0, "high": 830.0,
     "low": 810.0, "close": 825.0, "volume": 38000000, "scraped_at": "2024-06-02T10:00:00+00:00"},
    {"ticker": "NVDA", "date": "2024-06-03", "open": 825.0, "high": 840.0,
     "low": 820.0, "close": None,  "volume": 35000000, "scraped_at": "2024-06-03T10:00:00+00:00"},
]

SOURCE_C_FUND = [
    {
        "ticker": "NVDA", "company_name": "NVIDIA Corporation",
        "sector": "Technology", "market_cap": 1200000000000,
        "pe_ratio": 45.3, "forward_pe": 30.0, "price_to_book": 25.0,
        "fifty_two_week_high": 974.0, "fifty_two_week_low": 430.0,
        "fifty_day_avg": 850.0, "two_hundred_day_avg": 700.0,
        "analyst_target_price": 900.0,
        "scraped_at": "2024-06-01T10:00:00+00:00",
    }
]


# ---------------------------------------------------------------------------
# Helper to mock open() with JSON data
# ---------------------------------------------------------------------------

def make_mock_open(data: list) -> str:
    return json.dumps(data)


# ---------------------------------------------------------------------------
# Tests: Source A cleaning
# ---------------------------------------------------------------------------

class TestCleanSourceA:
    def test_deduplicates_by_url(self, tmp_path):
        p = tmp_path / "source_a_raw.json"
        p.write_text(json.dumps(SOURCE_A_DATA))
        df = clean_source_a(path=p)
        assert len(df) == 2  # 3 records, 1 duplicate removed

    def test_headline_stripped(self, tmp_path):
        p = tmp_path / "source_a_raw.json"
        p.write_text(json.dumps(SOURCE_A_DATA))
        df = clean_source_a(path=p)
        assert df["headline"].iloc[0] == df["headline"].iloc[0].strip()

    def test_published_at_is_datetime(self, tmp_path):
        p = tmp_path / "source_a_raw.json"
        p.write_text(json.dumps(SOURCE_A_DATA))
        df = clean_source_a(path=p)
        assert pd.api.types.is_datetime64_any_dtype(df["published_at"])

    def test_text_length_feature_added(self, tmp_path):
        p = tmp_path / "source_a_raw.json"
        p.write_text(json.dumps(SOURCE_A_DATA))
        df = clean_source_a(path=p)
        assert "text_length" in df.columns
        assert df["text_length"].iloc[0] > 0

    def test_is_multi_ticker_flag(self, tmp_path):
        p = tmp_path / "source_a_raw.json"
        p.write_text(json.dumps(SOURCE_A_DATA))
        df = clean_source_a(path=p)
        multi = df[df["is_multi_ticker"] == True]
        assert len(multi) >= 1


# ---------------------------------------------------------------------------
# Tests: Source B cleaning
# ---------------------------------------------------------------------------

class TestCleanSourceB:
    def test_sentiment_numeric_encoding(self, tmp_path):
        p = tmp_path / "source_b_raw.json"
        p.write_text(json.dumps(SOURCE_B_DATA))
        df = clean_source_b(path=p)
        assert df[df["ticker_sentiment_label"] == "bullish"]["ticker_sentiment_numeric"].iloc[0] == 1
        assert df[df["ticker_sentiment_label"] == "bearish"]["ticker_sentiment_numeric"].iloc[0] == -1

    def test_combined_score_calculated(self, tmp_path):
        p = tmp_path / "source_b_raw.json"
        p.write_text(json.dumps(SOURCE_B_DATA))
        df = clean_source_b(path=p)
        assert "combined_sentiment_score" in df.columns
        assert df["combined_sentiment_score"].notna().all()

    def test_published_at_parsed(self, tmp_path):
        p = tmp_path / "source_b_raw.json"
        p.write_text(json.dumps(SOURCE_B_DATA))
        df = clean_source_b(path=p)
        assert pd.api.types.is_datetime64_any_dtype(df["published_at"])

    def test_sentiment_score_map_complete(self):
        assert SENTIMENT_SCORE_MAP["bullish"] == 1
        assert SENTIMENT_SCORE_MAP["neutral"] == 0
        assert SENTIMENT_SCORE_MAP["bearish"] == -1


# ---------------------------------------------------------------------------
# Tests: Source C cleaning
# ---------------------------------------------------------------------------

class TestCleanSourceC:
    def test_drops_null_close(self, tmp_path):
        ohlcv_p = tmp_path / "source_c_raw.json"
        fund_p  = tmp_path / "source_c_fundamentals.json"
        ohlcv_p.write_text(json.dumps(SOURCE_C_OHLCV))
        fund_p.write_text(json.dumps(SOURCE_C_FUND))
        df, _ = clean_source_c(ohlcv_path=ohlcv_p, fund_path=fund_p)
        assert df["close"].isna().sum() == 0
        assert len(df) == 2  # 3 rows, 1 dropped for null close

    def test_daily_return_calculated(self, tmp_path):
        ohlcv_p = tmp_path / "source_c_raw.json"
        fund_p  = tmp_path / "source_c_fundamentals.json"
        ohlcv_p.write_text(json.dumps(SOURCE_C_OHLCV))
        fund_p.write_text(json.dumps(SOURCE_C_FUND))
        df, _ = clean_source_c(ohlcv_path=ohlcv_p, fund_path=fund_p)
        assert "daily_return" in df.columns

    def test_price_range_calculated(self, tmp_path):
        ohlcv_p = tmp_path / "source_c_raw.json"
        fund_p  = tmp_path / "source_c_fundamentals.json"
        ohlcv_p.write_text(json.dumps(SOURCE_C_OHLCV))
        fund_p.write_text(json.dumps(SOURCE_C_FUND))
        df, _ = clean_source_c(ohlcv_path=ohlcv_p, fund_path=fund_p)
        assert "price_range" in df.columns
        assert (df["price_range"] >= 0).all()

    def test_moving_averages_added(self, tmp_path):
        ohlcv_p = tmp_path / "source_c_raw.json"
        fund_p  = tmp_path / "source_c_fundamentals.json"
        ohlcv_p.write_text(json.dumps(SOURCE_C_OHLCV))
        fund_p.write_text(json.dumps(SOURCE_C_FUND))
        df, _ = clean_source_c(ohlcv_path=ohlcv_p, fund_path=fund_p)
        assert "ma_5" in df.columns
        assert "ma_20" in df.columns

    def test_fundamentals_returned(self, tmp_path):
        ohlcv_p = tmp_path / "source_c_raw.json"
        fund_p  = tmp_path / "source_c_fundamentals.json"
        ohlcv_p.write_text(json.dumps(SOURCE_C_OHLCV))
        fund_p.write_text(json.dumps(SOURCE_C_FUND))
        _, df_fund = clean_source_c(ohlcv_path=ohlcv_p, fund_path=fund_p)
        assert len(df_fund) == 1
        assert df_fund["ticker"].iloc[0] == "NVDA"