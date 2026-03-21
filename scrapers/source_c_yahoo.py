"""
Source C: Historical price data scraper using yfinance.

Why yfinance:
  - No API key or authentication required
  - Official Yahoo Finance data — OHLCV + fundamentals
  - Pulls up to 5 years of daily price history per ticker
  - Returns clean pandas DataFrames — minimal cleaning needed

Scraped fields per ticker:
  - date, open, high, low, close, volume, ticker
  - pe_ratio, market_cap, fifty_two_week_high, fifty_two_week_low (fundamentals)
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRACKED_TICKERS = [
    "NVDA", "MSFT", "GOOGL", "META", "AAPL",
    "AMZN", "AMD", "TSLA", "PLTR", "ORCL",
]

# How much history to pull
PERIOD  = "6mo"   # 6 months of daily data
INTERVAL = "1d"   # daily candles

RAW_OUTPUT         = Path(__file__).parent.parent / "data" / "raw" / "source_c_raw.json"
FUNDAMENTALS_OUTPUT = Path(__file__).parent.parent / "data" / "raw" / "source_c_fundamentals.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value) -> float | None:
    """Convert a value to float safely — returns None if not possible."""
    try:
        f = float(value)
        return None if pd.isna(f) else round(f, 4)
    except (TypeError, ValueError):
        return None


def _extract_fundamentals(ticker_obj: yf.Ticker, ticker: str) -> dict:
    """
    Pull key fundamental metrics from yfinance info dict.
    These are snapshot values (not historical) — stored separately.
    """
    try:
        info = ticker_obj.info
    except Exception:
        info = {}

    return {
        "ticker":               ticker,
        "company_name":         info.get("longName", ticker),
        "sector":               info.get("sector", "Technology"),
        "market_cap":           _safe_float(info.get("marketCap")),
        "pe_ratio":             _safe_float(info.get("trailingPE")),
        "forward_pe":           _safe_float(info.get("forwardPE")),
        "price_to_book":        _safe_float(info.get("priceToBook")),
        "fifty_two_week_high":  _safe_float(info.get("fiftyTwoWeekHigh")),
        "fifty_two_week_low":   _safe_float(info.get("fiftyTwoWeekLow")),
        "fifty_day_avg":        _safe_float(info.get("fiftyDayAverage")),
        "two_hundred_day_avg":  _safe_float(info.get("twoHundredDayAverage")),
        "analyst_target_price": _safe_float(info.get("targetMeanPrice")),
        "scraped_at":           datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Core scraper
# ---------------------------------------------------------------------------

class YahooFinanceScraper:
    """
    Pulls historical OHLCV price data and fundamental metrics via yfinance.

    Two outputs:
      1. source_c_raw.json        — daily OHLCV records per ticker
      2. source_c_fundamentals.json — snapshot fundamentals per ticker
    """

    def __init__(self, tickers: list[str] = TRACKED_TICKERS, period: str = PERIOD):
        self.tickers = tickers
        self.period  = period

    def _fetch_ohlcv(self, ticker: str) -> list[dict]:
        """Download daily OHLCV history and convert to list of dicts."""
        try:
            logger.info(f"[{ticker}] Downloading {self.period} of price history")
            df = yf.download(
                ticker,
                period=self.period,
                interval=INTERVAL,
                auto_adjust=True,   # adjusts for splits + dividends
                progress=False,     # suppress yfinance download bar
            )

            if df.empty:
                logger.warning(f"[{ticker}] No price data returned")
                return []

            # Flatten MultiIndex columns if present (yfinance quirk for single ticker)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            records = []
            for date, row in df.iterrows():
                records.append({
                    "ticker":    ticker,
                    "date":      date.strftime("%Y-%m-%d"),
                    "open":      _safe_float(row.get("Open")),
                    "high":      _safe_float(row.get("High")),
                    "low":       _safe_float(row.get("Low")),
                    "close":     _safe_float(row.get("Close")),
                    "volume":    int(row.get("Volume", 0)),
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                })

            logger.info(f"[{ticker}] {len(records)} daily records extracted")
            return records

        except Exception as exc:
            logger.error(f"[{ticker}] OHLCV fetch failed: {exc}")
            return []

    def _fetch_fundamentals(self, ticker: str) -> dict:
        """Pull fundamental snapshot for a single ticker."""
        try:
            ticker_obj = yf.Ticker(ticker)
            fundamentals = _extract_fundamentals(ticker_obj, ticker)
            logger.info(
                f"[{ticker}] Fundamentals: "
                f"P/E={fundamentals['pe_ratio']} | "
                f"MarketCap={fundamentals['market_cap']}"
            )
            return fundamentals
        except Exception as exc:
            logger.error(f"[{ticker}] Fundamentals fetch failed: {exc}")
            return {"ticker": ticker, "error": str(exc)}

    def scrape(self) -> tuple[list[dict], list[dict]]:
        """
        Scrape all tickers.
        Returns (ohlcv_records, fundamentals_list).
        """
        all_ohlcv        = []
        all_fundamentals = []

        for ticker in self.tickers:
            ohlcv        = self._fetch_ohlcv(ticker)
            fundamentals = self._fetch_fundamentals(ticker)

            all_ohlcv.extend(ohlcv)
            all_fundamentals.append(fundamentals)

        logger.success(
            f"Source C complete: "
            f"{len(all_ohlcv)} OHLCV records | "
            f"{len(all_fundamentals)} ticker fundamentals"
        )
        return all_ohlcv, all_fundamentals


# ---------------------------------------------------------------------------
# Save + Run
# ---------------------------------------------------------------------------

def save_raw(
    ohlcv: list[dict],
    fundamentals: list[dict],
    ohlcv_path: Path = RAW_OUTPUT,
    fund_path: Path = FUNDAMENTALS_OUTPUT,
) -> None:
    ohlcv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(ohlcv_path, "w", encoding="utf-8") as f:
        json.dump(ohlcv, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(ohlcv)} OHLCV records → {ohlcv_path}")

    with open(fund_path, "w", encoding="utf-8") as f:
        json.dump(fundamentals, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(fundamentals)} fundamentals → {fund_path}")


def run(tickers: list[str] = TRACKED_TICKERS, period: str = PERIOD) -> tuple[list[dict], list[dict]]:
    scraper              = YahooFinanceScraper(tickers=tickers, period=period)
    ohlcv, fundamentals  = scraper.scrape()
    save_raw(ohlcv, fundamentals)
    return ohlcv, fundamentals


if __name__ == "__main__":
    run()