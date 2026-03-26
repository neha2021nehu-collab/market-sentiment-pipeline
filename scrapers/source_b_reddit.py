"""
Source B: Social & News Sentiment scraper using Alpha Vantage News API.

Why Alpha Vantage over Reddit/StockTwits:
  - Free API key, instant approval, no waitlists
  - Returns news articles with pre-computed sentiment scores per ticker
  - Aggregates from 50+ sources including financial news + social signals
  - Sentiment labels (Bullish/Bearish/Neutral) + confidence scores built in

Endpoint: https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={TICKER}&apikey={KEY}

Scraped fields per article:
  - title, summary, url, source, published_at, ticker,
    overall_sentiment_label, overall_sentiment_score,
    ticker_sentiment_label, ticker_sentiment_score
"""

import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")
BASE_URL = "https://www.alphavantage.co/query"

TRACKED_TICKERS = [
    "NVDA", "MSFT", "GOOGL", "META", "AAPL",
    "AMZN", "AMD", "TSLA", "PLTR", "ORCL",
]

RAW_OUTPUT = Path(__file__).parent.parent / "data" / "raw" / "source_b_raw.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept":     "application/json",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_timestamp(raw: str | None) -> str:
    """
    Alpha Vantage uses format: '20240601T103000'
    Convert to ISO-8601 UTC.
    """
    if not raw:
        return datetime.now(timezone.utc).isoformat()
    try:
        # Format: YYYYMMDDTHHmmSS
        dt = datetime.strptime(raw, "%Y%m%dT%H%M%S")
        return dt.replace(tzinfo=timezone.utc).isoformat()
    except ValueError:
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).isoformat()
        except ValueError:
            return datetime.now(timezone.utc).isoformat()


def _normalize_label(label: str | None) -> str:
    """Normalize sentiment label to lowercase simple form."""
    if not label:
        return "neutral"
    label = label.lower()
    if "bullish" in label:
        return "bullish"
    if "bearish" in label:
        return "bearish"
    return "neutral"


def _random_delay(min_s: float = 1.5, max_s: float = 3.5) -> float:
    return random.uniform(min_s, max_s)


# ---------------------------------------------------------------------------
# Core scraper
# ---------------------------------------------------------------------------

class AlphaVantageSentimentScraper:
    """
    Fetches news + sentiment data from Alpha Vantage News Sentiment API.

    Free tier: 25 requests/day — we batch multiple tickers per request
    to stay well within limits (1 request per ticker = 10 requests total).

    Each article includes:
      - Overall sentiment for the article
      - Per-ticker sentiment score specific to that stock
    """

    def __init__(self, tickers: list[str] = TRACKED_TICKERS, max_per_ticker: int = 20):
        self.tickers        = tickers
        self.max_per_ticker = max_per_ticker
        self.session        = requests.Session()
        self.session.headers.update(HEADERS)

        # Read fresh at init so GitHub Actions env vars are always picked up
        self.api_key = os.environ.get("ALPHAVANTAGE_API_KEY") or API_KEY
        if not self.api_key:
            raise ValueError(
                "ALPHAVANTAGE_API_KEY not found — "
                "set it in .env locally or as a GitHub Actions secret"
            )

    def _fetch_ticker(self, ticker: str) -> list[dict]:
        """Fetch news sentiment articles for a single ticker."""
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers":  ticker,
            "limit":    self.max_per_ticker,
            "apikey":   self.api_key,
        }

        try:
            response = self.session.get(BASE_URL, params=params, timeout=15)

            if response.status_code != 200:
                logger.warning(f"[{ticker}] HTTP {response.status_code}")
                return []

            data = response.json()

            # Alpha Vantage returns error messages in the response body
            if "Information" in data:
                logger.error(f"API limit hit: {data['Information']}")
                return []

            if "Note" in data:
                logger.warning(f"API note: {data['Note']}")

            articles_raw = data.get("feed", [])
            logger.info(f"[{ticker}] {len(articles_raw)} articles fetched")

            articles = []
            for item in articles_raw:
                # Extract per-ticker sentiment from the ticker_sentiments list
                ticker_sentiment_label = "neutral"
                ticker_sentiment_score = 0.0
                for ts in item.get("ticker_sentiment", []):
                    if ts.get("ticker") == ticker:
                        ticker_sentiment_label = _normalize_label(
                            ts.get("ticker_sentiment_label")
                        )
                        try:
                            ticker_sentiment_score = float(
                                ts.get("ticker_sentiment_score", 0)
                            )
                        except (ValueError, TypeError):
                            ticker_sentiment_score = 0.0
                        break

                articles.append({
                    "title":                   item.get("title", "").strip(),
                    "summary":                 item.get("summary", "").strip()[:500],
                    "url":                     item.get("url", ""),
                    "source":                  item.get("source", ""),
                    "published_at":            _parse_timestamp(item.get("time_published")),
                    "ticker":                  ticker,
                    "overall_sentiment_label": _normalize_label(
                                                   item.get("overall_sentiment_label")
                                               ),
                    "overall_sentiment_score": float(
                                                   item.get("overall_sentiment_score", 0)
                                               ),
                    "ticker_sentiment_label":  ticker_sentiment_label,
                    "ticker_sentiment_score":  ticker_sentiment_score,
                    "data_source":             "alphavantage_news_sentiment",
                    "scraped_at":              datetime.now(timezone.utc).isoformat(),
                })

            return articles

        except requests.exceptions.Timeout:
            logger.error(f"[{ticker}] Request timed out")
        except requests.exceptions.ConnectionError:
            logger.error(f"[{ticker}] Connection error")
        except Exception as exc:
            logger.error(f"[{ticker}] Unexpected error: {exc}")

        return []

    def scrape(self) -> list[dict]:
        """Scrape all tickers with polite delays between requests."""
        all_articles = []

        for i, ticker in enumerate(self.tickers):
            logger.info(f"Fetching sentiment {i+1}/{len(self.tickers)}: {ticker}")
            articles = self._fetch_ticker(ticker)
            all_articles.extend(articles)

            if i < len(self.tickers) - 1:
                delay = _random_delay()
                logger.debug(f"Waiting {delay:.1f}s...")
                time.sleep(delay)

        # Summary stats
        bullish = sum(1 for a in all_articles if a["ticker_sentiment_label"] == "bullish")
        bearish = sum(1 for a in all_articles if a["ticker_sentiment_label"] == "bearish")
        neutral = sum(1 for a in all_articles if a["ticker_sentiment_label"] == "neutral")

        logger.success(
            f"Source B complete: {len(all_articles)} articles | "
            f"Bullish: {bullish} | Bearish: {bearish} | Neutral: {neutral}"
        )
        return all_articles


# ---------------------------------------------------------------------------
# Save + Run
# ---------------------------------------------------------------------------

def save_raw(articles: list[dict], path: Path = RAW_OUTPUT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(articles)} articles → {path}")


def run(tickers: list[str] = TRACKED_TICKERS, max_per_ticker: int = 20) -> list[dict]:
    scraper  = AlphaVantageSentimentScraper(tickers=tickers, max_per_ticker=max_per_ticker)
    articles = scraper.scrape()
    save_raw(articles)
    return articles


if __name__ == "__main__":
    run()