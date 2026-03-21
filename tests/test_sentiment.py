"""
Unit tests for analysis/sentiment.py
Run: pytest tests/test_sentiment.py -v
"""

import pytest
import pandas as pd
from analysis.sentiment import (
    compound_to_label,
    VADERScorer,
    build_daily_sentiment,
    BULLISH_THRESHOLD,
    BEARISH_THRESHOLD,
)


class TestCompoundToLabel:
    def test_bullish_threshold(self):
        assert compound_to_label(0.05) == "bullish"
        assert compound_to_label(0.8)  == "bullish"

    def test_bearish_threshold(self):
        assert compound_to_label(-0.05) == "bearish"
        assert compound_to_label(-0.9)  == "bearish"

    def test_neutral_zone(self):
        assert compound_to_label(0.0)   == "neutral"
        assert compound_to_label(0.04)  == "neutral"
        assert compound_to_label(-0.04) == "neutral"

    def test_thresholds_are_symmetric(self):
        assert BULLISH_THRESHOLD == abs(BEARISH_THRESHOLD)


class TestVADERScorer:
    def setup_method(self):
        self.scorer = VADERScorer()

    def test_positive_headline(self):
        result = self.scorer.score_text("NVDA reports excellent earnings, beating expectations!")
        assert result["compound"] > 0

    def test_negative_headline(self):
        result = self.scorer.score_text("MSFT crashes amid massive layoffs and losses")
        assert result["compound"] < 0

    def test_empty_text_returns_zero(self):
        result = self.scorer.score_text("")
        assert result["compound"] == 0.0

    def test_score_article_returns_all_fields(self):
        result = self.scorer.score_article(
            "NVDA beats earnings expectations",
            "Nvidia reported strong quarterly results"
        )
        assert "vader_compound" in result
        assert "vader_label" in result
        assert "vader_headline_compound" in result
        assert "vader_summary_compound" in result
        assert "vader_positive" in result
        assert "vader_negative" in result

    def test_score_article_label_consistent_with_compound(self):
        result = self.scorer.score_article(
            "Catastrophic collapse in tech stocks",
            "Markets crash as investors panic sell"
        )
        if result["vader_compound"] >= 0.05:
            assert result["vader_label"] == "bullish"
        elif result["vader_compound"] <= -0.05:
            assert result["vader_label"] == "bearish"
        else:
            assert result["vader_label"] == "neutral"

    def test_score_dataframe(self):
        df = pd.DataFrame([
            {
                "headline": "NVDA surges on AI demand",
                "summary":  "Strong earnings reported",
                "published_at": "2024-06-01T10:00:00+00:00",
                "tickers_mentioned": ["NVDA"],
                "source": "techcrunch_ai",
            },
            {
                "headline": "AMD faces headwinds",
                "summary":  "Competition intensifies",
                "published_at": "2024-06-02T10:00:00+00:00",
                "tickers_mentioned": ["AMD"],
                "source": "ars_technica",
            },
        ])
        df_scored = self.scorer.score_dataframe(df)
        assert "vader_compound" in df_scored.columns
        assert "vader_label" in df_scored.columns
        assert len(df_scored) == 2


class TestBuildDailySentiment:
    def test_aggregates_by_ticker_and_date(self):
        df = pd.DataFrame([
            {
                "headline": "NVDA strong",
                "summary": "Good results",
                "published_at": pd.Timestamp("2024-06-01", tz="UTC"),
                "tickers_mentioned": ["NVDA"],
                "vader_compound": 0.5,
                "vader_label": "bullish",
            },
            {
                "headline": "NVDA faces risk",
                "summary": "Concerns rise",
                "published_at": pd.Timestamp("2024-06-01", tz="UTC"),
                "tickers_mentioned": ["NVDA"],
                "vader_compound": -0.2,
                "vader_label": "bearish",
            },
        ])
        daily = build_daily_sentiment(df)
        assert len(daily) == 1
        assert daily.iloc[0]["ticker"] == "NVDA"
        assert daily.iloc[0]["article_count"] == 2
        assert "avg_vader_score" in daily.columns
        assert "dominant_label" in daily.columns

    def test_empty_tickers_excluded(self):
        df = pd.DataFrame([
            {
                "headline": "General tech news",
                "summary": "No specific ticker",
                "published_at": pd.Timestamp("2024-06-01", tz="UTC"),
                "tickers_mentioned": [],
                "vader_compound": 0.3,
                "vader_label": "bullish",
            }
        ])
        daily = build_daily_sentiment(df)
        assert daily.empty