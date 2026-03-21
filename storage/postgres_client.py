"""
PostgreSQL storage client for structured price data.

Tables:
  - ohlcv_prices    — daily OHLCV + technical indicators per ticker
  - fundamentals    — snapshot fundamental metrics per ticker
  - sentiment_scores — daily aggregated sentiment scores per ticker

Why PostgreSQL for this data:
  - Price data is strictly structured — same columns every time
  - SQL makes time-series queries easy (date ranges, rolling averages)
  - JOIN between price and sentiment tables powers the core insight
  - UNIQUE constraints prevent duplicate price records
"""

import os
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import (
    create_engine, text,
    Column, String, Float, Integer, Date, DateTime, Boolean,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, Session

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

POSTGRES_URI = os.getenv("POSTGRES_URI", "")
Base = declarative_base()

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class OHLCVPrice(Base):
    __tablename__ = "ohlcv_prices"
    __table_args__ = (UniqueConstraint("ticker", "date", name="uq_ticker_date"),)

    id           = Column(Integer, primary_key=True, autoincrement=True)
    ticker       = Column(String(10),  nullable=False, index=True)
    date         = Column(Date,        nullable=False, index=True)
    open         = Column(Float)
    high         = Column(Float)
    low          = Column(Float)
    close        = Column(Float,       nullable=False)
    volume       = Column(Integer)
    daily_return = Column(Float)
    price_range  = Column(Float)
    ma_5         = Column(Float)
    ma_20        = Column(Float)
    scraped_at   = Column(DateTime)


class Fundamental(Base):
    __tablename__ = "fundamentals"
    __table_args__ = (UniqueConstraint("ticker", name="uq_fund_ticker"),)

    id                   = Column(Integer, primary_key=True, autoincrement=True)
    ticker               = Column(String(10), nullable=False, index=True)
    company_name         = Column(String(200))
    sector               = Column(String(100))
    market_cap           = Column(Float)
    pe_ratio             = Column(Float)
    forward_pe           = Column(Float)
    price_to_book        = Column(Float)
    fifty_two_week_high  = Column(Float)
    fifty_two_week_low   = Column(Float)
    fifty_day_avg        = Column(Float)
    two_hundred_day_avg  = Column(Float)
    analyst_target_price = Column(Float)
    scraped_at           = Column(DateTime)


class SentimentScore(Base):
    __tablename__ = "sentiment_scores"
    __table_args__ = (UniqueConstraint("ticker", "date", name="uq_sentiment_ticker_date"),)

    id               = Column(Integer, primary_key=True, autoincrement=True)
    ticker           = Column(String(10), nullable=False, index=True)
    date             = Column(Date,       nullable=False, index=True)
    article_count    = Column(Integer)
    avg_vader_score  = Column(Float)
    bullish_count    = Column(Integer)
    bearish_count    = Column(Integer)
    neutral_count    = Column(Integer)
    dominant_label   = Column(String(10))


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class PostgresStorage:
    """
    Handles all PostgreSQL operations for structured price + sentiment data.
    Uses INSERT ... ON CONFLICT DO NOTHING for safe upserts.
    """

    def __init__(self):
        if not POSTGRES_URI:
            raise ValueError("POSTGRES_URI not found in .env")

        self.engine = create_engine(POSTGRES_URI, pool_pre_ping=True)
        Base.metadata.create_all(self.engine)
        logger.info("PostgreSQL connected — tables verified")

    def _safe_value(self, v):
        """Convert numpy/pandas types to Python natives for SQLAlchemy."""
        if pd.isna(v) if not isinstance(v, (list, dict)) else False:
            return None
        if hasattr(v, "item"):
            return v.item()
        if hasattr(v, "date") and callable(v.date):
            return v.date()
        return v

    def insert_ohlcv(self, df: pd.DataFrame) -> int:
        """Insert OHLCV price records — skip duplicates."""
        if df.empty:
            return 0

        inserted = 0
        with Session(self.engine) as session:
            for _, row in df.iterrows():
                stmt = text("""
                    INSERT INTO ohlcv_prices
                        (ticker, date, open, high, low, close, volume,
                         daily_return, price_range, ma_5, ma_20, scraped_at)
                    VALUES
                        (:ticker, :date, :open, :high, :low, :close, :volume,
                         :daily_return, :price_range, :ma_5, :ma_20, :scraped_at)
                    ON CONFLICT (ticker, date) DO NOTHING
                """)
                session.execute(stmt, {
                    "ticker":       str(row["ticker"]),
                    "date":         pd.Timestamp(row["date"]).date(),
                    "open":         self._safe_value(row.get("open")),
                    "high":         self._safe_value(row.get("high")),
                    "low":          self._safe_value(row.get("low")),
                    "close":        self._safe_value(row.get("close")),
                    "volume":       int(row.get("volume", 0)),
                    "daily_return": self._safe_value(row.get("daily_return")),
                    "price_range":  self._safe_value(row.get("price_range")),
                    "ma_5":         self._safe_value(row.get("ma_5")),
                    "ma_20":        self._safe_value(row.get("ma_20")),
                    "scraped_at":   datetime.now(timezone.utc),
                })
                inserted += 1
            session.commit()

        logger.success(f"PostgreSQL ohlcv_prices: {inserted} rows inserted")
        return inserted

    def insert_fundamentals(self, df: pd.DataFrame) -> int:
        """Insert/update fundamental metrics per ticker."""
        if df.empty:
            return 0

        inserted = 0
        with Session(self.engine) as session:
            for _, row in df.iterrows():
                stmt = text("""
                    INSERT INTO fundamentals
                        (ticker, company_name, sector, market_cap, pe_ratio,
                         forward_pe, price_to_book, fifty_two_week_high,
                         fifty_two_week_low, fifty_day_avg, two_hundred_day_avg,
                         analyst_target_price, scraped_at)
                    VALUES
                        (:ticker, :company_name, :sector, :market_cap, :pe_ratio,
                         :forward_pe, :price_to_book, :fifty_two_week_high,
                         :fifty_two_week_low, :fifty_day_avg, :two_hundred_day_avg,
                         :analyst_target_price, :scraped_at)
                    ON CONFLICT (ticker) DO UPDATE SET
                        pe_ratio             = EXCLUDED.pe_ratio,
                        market_cap           = EXCLUDED.market_cap,
                        analyst_target_price = EXCLUDED.analyst_target_price,
                        scraped_at           = EXCLUDED.scraped_at
                """)
                session.execute(stmt, {
                    "ticker":               str(row["ticker"]),
                    "company_name":         str(row.get("company_name", "")),
                    "sector":               str(row.get("sector", "Technology")),
                    "market_cap":           self._safe_value(row.get("market_cap")),
                    "pe_ratio":             self._safe_value(row.get("pe_ratio")),
                    "forward_pe":           self._safe_value(row.get("forward_pe")),
                    "price_to_book":        self._safe_value(row.get("price_to_book")),
                    "fifty_two_week_high":  self._safe_value(row.get("fifty_two_week_high")),
                    "fifty_two_week_low":   self._safe_value(row.get("fifty_two_week_low")),
                    "fifty_day_avg":        self._safe_value(row.get("fifty_day_avg")),
                    "two_hundred_day_avg":  self._safe_value(row.get("two_hundred_day_avg")),
                    "analyst_target_price": self._safe_value(row.get("analyst_target_price")),
                    "scraped_at":           datetime.now(timezone.utc),
                })
                inserted += 1
            session.commit()

        logger.success(f"PostgreSQL fundamentals: {inserted} rows inserted")
        return inserted

    def insert_sentiment_scores(self, df: pd.DataFrame) -> int:
        """Insert daily sentiment scores per ticker."""
        if df.empty:
            return 0

        inserted = 0
        with Session(self.engine) as session:
            for _, row in df.iterrows():
                stmt = text("""
                    INSERT INTO sentiment_scores
                        (ticker, date, article_count, avg_vader_score,
                         bullish_count, bearish_count, neutral_count, dominant_label)
                    VALUES
                        (:ticker, :date, :article_count, :avg_vader_score,
                         :bullish_count, :bearish_count, :neutral_count, :dominant_label)
                    ON CONFLICT (ticker, date) DO UPDATE SET
                        avg_vader_score = EXCLUDED.avg_vader_score,
                        dominant_label  = EXCLUDED.dominant_label
                """)
                session.execute(stmt, {
                    "ticker":          str(row["ticker"]),
                    "date":            pd.Timestamp(str(row["date"])).date(),
                    "article_count":   int(row.get("article_count", 0)),
                    "avg_vader_score": self._safe_value(row.get("avg_vader_score")),
                    "bullish_count":   int(row.get("bullish_count", 0)),
                    "bearish_count":   int(row.get("bearish_count", 0)),
                    "neutral_count":   int(row.get("neutral_count", 0)),
                    "dominant_label":  str(row.get("dominant_label", "neutral")),
                })
                inserted += 1
            session.commit()

        logger.success(f"PostgreSQL sentiment_scores: {inserted} rows inserted")
        return inserted

    def query_sentiment_vs_price(self, ticker: str) -> pd.DataFrame:
        """
        The core insight query — join price data with sentiment scores.
        'Is the news saying Buy while price is dropping?'
        """
        query = text("""
            SELECT
                o.ticker,
                o.date,
                o.close,
                o.daily_return,
                s.avg_vader_score,
                s.dominant_label,
                s.article_count
            FROM ohlcv_prices o
            LEFT JOIN sentiment_scores s
                ON o.ticker = s.ticker AND o.date = s.date
            WHERE o.ticker = :ticker
            ORDER BY o.date DESC
            LIMIT 30
        """)
        with Session(self.engine) as session:
            result = session.execute(query, {"ticker": ticker})
            return pd.DataFrame(result.fetchall(), columns=result.keys())

    def close(self) -> None:
        self.engine.dispose()
        logger.info("PostgreSQL connection closed")


def run(df_ohlcv=None, df_fundamentals=None, df_daily_sentiment=None) -> dict:
    """Insert all structured data into PostgreSQL."""
    storage = PostgresStorage()
    results = {}

    try:
        if df_ohlcv is not None and not df_ohlcv.empty:
            results["ohlcv"] = storage.insert_ohlcv(df_ohlcv)

        if df_fundamentals is not None and not df_fundamentals.empty:
            results["fundamentals"] = storage.insert_fundamentals(df_fundamentals)

        if df_daily_sentiment is not None and not df_daily_sentiment.empty:
            results["sentiment_scores"] = storage.insert_sentiment_scores(df_daily_sentiment)
    finally:
        storage.close()

    return results