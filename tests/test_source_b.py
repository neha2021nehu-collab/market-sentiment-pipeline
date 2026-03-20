"""
Unit tests for source_b_reddit.py (Alpha Vantage sentiment scraper).
All offline — no real HTTP calls made.
Run: pytest tests/test_source_b.py -v
"""

import pytest
from unittest.mock import patch, MagicMock
from scrapers.source_b_reddit import (
    _parse_timestamp,
    _normalize_label,
    _random_delay,
    AlphaVantageSentimentScraper,
)


class TestParseTimestamp:
    def test_alphavantage_format(self):
        result = _parse_timestamp("20240601T103000")
        assert "2024-06-01" in result
        assert "T" in result

    def test_none_falls_back_to_now(self):
        assert "T" in _parse_timestamp(None)

    def test_invalid_falls_back(self):
        assert "T" in _parse_timestamp("bad-date")


class TestNormalizeLabel:
    def test_bullish(self):
        assert _normalize_label("Bullish") == "bullish"

    def test_somewhat_bullish(self):
        assert _normalize_label("Somewhat-Bullish") == "bullish"

    def test_bearish(self):
        assert _normalize_label("Bearish") == "bearish"

    def test_somewhat_bearish(self):
        assert _normalize_label("Somewhat-Bearish") == "bearish"

    def test_neutral(self):
        assert _normalize_label("Neutral") == "neutral"

    def test_none_returns_neutral(self):
        assert _normalize_label(None) == "neutral"


class TestRandomDelay:
    def test_within_default_bounds(self):
        for _ in range(30):
            assert 1.5 <= _random_delay() <= 3.5


class TestAlphaVantageScraper:
    def test_init_success(self):
        with patch("scrapers.source_b_reddit.API_KEY", "test_key_123"):
            scraper = AlphaVantageSentimentScraper(tickers=["NVDA"], max_per_ticker=5)
        assert "NVDA" in scraper.tickers
        assert scraper.max_per_ticker == 5

    def test_init_no_key_raises(self):
        with patch.dict("os.environ", {"ALPHAVANTAGE_API_KEY": ""}):
            with patch("scrapers.source_b_reddit.API_KEY", ""):
                with pytest.raises(ValueError, match="ALPHAVANTAGE_API_KEY"):
                    AlphaVantageSentimentScraper()

    @patch.dict("os.environ", {"ALPHAVANTAGE_API_KEY": "test_key_123"})
    @patch("scrapers.source_b_reddit.requests.Session.get")
    def test_fetch_ticker_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "feed": [
                {
                    "title":                   "NVDA beats earnings expectations",
                    "summary":                 "Nvidia reported strong Q2 results...",
                    "url":                     "https://example.com/nvda",
                    "source":                  "TechCrunch",
                    "time_published":          "20240601T103000",
                    "overall_sentiment_label": "Bullish",
                    "overall_sentiment_score": "0.35",
                    "ticker_sentiment": [
                        {
                            "ticker":                  "NVDA",
                            "ticker_sentiment_label":  "Bullish",
                            "ticker_sentiment_score":  "0.42",
                        }
                    ],
                }
            ]
        }
        mock_get.return_value = mock_response

        with patch("scrapers.source_b_reddit.API_KEY", "test_key_123"):
            scraper  = AlphaVantageSentimentScraper(tickers=["NVDA"], max_per_ticker=5)
            articles = scraper._fetch_ticker("NVDA")

        assert len(articles) == 1
        assert articles[0]["ticker"] == "NVDA"
        assert articles[0]["ticker_sentiment_label"] == "bullish"
        assert articles[0]["overall_sentiment_label"] == "bullish"
        assert articles[0]["data_source"] == "alphavantage_news_sentiment"

    @patch.dict("os.environ", {"ALPHAVANTAGE_API_KEY": "test_key_123"})
    @patch("scrapers.source_b_reddit.requests.Session.get")
    def test_api_limit_returns_empty(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "Information": "API rate limit reached."
        }
        mock_get.return_value = mock_response

        with patch("scrapers.source_b_reddit.API_KEY", "test_key_123"):
            scraper  = AlphaVantageSentimentScraper(tickers=["NVDA"], max_per_ticker=5)
            articles = scraper._fetch_ticker("NVDA")

        assert articles == []