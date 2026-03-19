"""
Unit tests for source_a_playwright.py — all offline, no browser launched.
Run: pytest tests/test_source_a.py -v
"""

import pytest
from scrapers.source_a_playwright import (
    _extract_tickers,
    _parse_timestamp,
    _random_delay,
    _struct_time_to_iso,
)
import time


class TestExtractTickers:
    def test_finds_single_ticker(self):
        assert "NVDA" in _extract_tickers("NVDA reports record GPU sales")

    def test_finds_multiple_tickers(self):
        result = _extract_tickers("MSFT and GOOGL both beat earnings")
        assert "MSFT" in result
        assert "GOOGL" in result

    def test_lowercase_no_match(self):
        assert "AI" not in _extract_tickers("This is a regular ai discussion")

    def test_no_tickers_returns_empty(self):
        assert _extract_tickers("The weather today is nice") == []

    def test_partial_word_no_match(self):
        assert "AI" not in _extract_tickers("Try AGAIN tomorrow")

    def test_ticker_in_summary_detected(self):
        assert "TSLA" in _extract_tickers("Some article about cars... TSLA surges 10%")


class TestParseTimestamp:
    def test_valid_iso_with_z(self):
        result = _parse_timestamp("2024-06-01T10:30:00Z")
        assert result.startswith("2024-06-01")

    def test_valid_iso_with_offset(self):
        assert "2024-06-01" in _parse_timestamp("2024-06-01T10:30:00+05:30")

    def test_none_returns_current_time(self):
        assert "T" in _parse_timestamp(None)

    def test_invalid_falls_back_to_now(self):
        assert "T" in _parse_timestamp("not-a-date")


class TestStructTimeToIso:
    def test_converts_struct_time(self):
        st = time.strptime("2024-06-01 10:30:00", "%Y-%m-%d %H:%M:%S")
        result = _struct_time_to_iso(st)
        assert "2024-06-01" in result
        assert "T" in result

    def test_bad_input_falls_back(self):
        assert "T" in _struct_time_to_iso(None)


class TestRandomDelay:
    def test_within_custom_bounds(self):
        for _ in range(50):
            assert 1.0 <= _random_delay(1.0, 3.0) <= 3.0

    def test_default_bounds(self):
        assert 2.5 <= _random_delay() <= 6.0