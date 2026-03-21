"""
MongoDB storage client for unstructured text data.

Collections:
  - news_articles      — Source A cleaned news (with VADER scores)
  - sentiment_articles — Source B Alpha Vantage sentiment articles
  - daily_sentiment    — Aggregated daily sentiment per ticker
"""

import os
from datetime import datetime, timezone

import numpy as np
from dotenv import load_dotenv
from loguru import logger
from pymongo import MongoClient, ASCENDING, DESCENDING

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "")
MONGO_DB  = os.getenv("MONGO_DB", "sentiment_pulse")

COLLECTIONS = {
    "news":            "news_articles",
    "sentiment":       "sentiment_articles",
    "daily_sentiment": "daily_sentiment",
}


class MongoStorage:
    def __init__(self):
        if not MONGO_URI:
            raise ValueError("MONGO_URI not found in .env")
        self.client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10_000)
        self.db     = self.client[MONGO_DB]
        self._ensure_indexes()
        logger.info(f"MongoDB connected → {MONGO_DB}")

    def _ensure_indexes(self) -> None:
        self.db[COLLECTIONS["news"]].create_index("url", unique=True)
        self.db[COLLECTIONS["news"]].create_index(
            [("tickers_mentioned", ASCENDING), ("published_at", DESCENDING)]
        )
        self.db[COLLECTIONS["sentiment"]].create_index(
            [("ticker", ASCENDING), ("published_at", DESCENDING)]
        )
        self.db[COLLECTIONS["daily_sentiment"]].create_index(
            [("ticker", ASCENDING), ("date", DESCENDING)], unique=True
        )
        logger.info("MongoDB indexes verified")

    def _sanitize(self, record: dict) -> dict:
        """
        Recursively convert numpy/pandas types to Python natives.
        Handles scalars, arrays, lists, and timestamps.
        """
        sanitized = {}
        for k, v in record.items():
            sanitized[k] = self._convert(v)
        return sanitized

    def _convert(self, v):
        """Convert a single value to a MongoDB-safe Python type."""
        # numpy integer types
        if isinstance(v, (np.integer,)):
            return int(v)
        # numpy float types
        if isinstance(v, (np.floating,)):
            return None if np.isnan(v) else float(v)
        # numpy bool
        if isinstance(v, np.bool_):
            return bool(v)
        # numpy arrays → list
        if isinstance(v, np.ndarray):
            return [self._convert(i) for i in v.tolist()]
        # pandas Timestamp or datetime with isoformat
        if hasattr(v, "isoformat"):
            return v.isoformat()
        # plain list — recurse
        if isinstance(v, list):
            return [self._convert(i) for i in v]
        # NaN float
        if isinstance(v, float) and np.isnan(v):
            return None
        return v

    def upsert_news(self, records: list[dict]) -> dict:
        inserted = skipped = 0
        collection = self.db[COLLECTIONS["news"]]
        for record in records:
            doc = self._sanitize(record)
            try:
                collection.update_one(
                    {"url": doc["url"]},
                    {"$setOnInsert": doc},
                    upsert=True,
                )
                inserted += 1
            except Exception:
                skipped += 1
        logger.success(f"MongoDB news_articles: {inserted} upserted | {skipped} skipped")
        return {"inserted": inserted, "skipped": skipped}

    def upsert_sentiment(self, records: list[dict]) -> dict:
        inserted = skipped = 0
        collection = self.db[COLLECTIONS["sentiment"]]
        for record in records:
            doc = self._sanitize(record)
            try:
                collection.update_one(
                    {"url": doc["url"], "ticker": doc["ticker"]},
                    {"$setOnInsert": doc},
                    upsert=True,
                )
                inserted += 1
            except Exception:
                skipped += 1
        logger.success(f"MongoDB sentiment_articles: {inserted} upserted | {skipped} skipped")
        return {"inserted": inserted, "skipped": skipped}

    def upsert_daily_sentiment(self, records: list[dict]) -> dict:
        inserted = skipped = 0
        collection = self.db[COLLECTIONS["daily_sentiment"]]
        for record in records:
            doc = self._sanitize(record)
            try:
                collection.update_one(
                    {"ticker": doc["ticker"], "date": str(doc["date"])},
                    {"$set": doc},
                    upsert=True,
                )
                inserted += 1
            except Exception:
                skipped += 1
        logger.success(f"MongoDB daily_sentiment: {inserted} upserted | {skipped} skipped")
        return {"inserted": inserted, "skipped": skipped}

    def close(self) -> None:
        self.client.close()
        logger.info("MongoDB connection closed")


def run(df_news=None, df_sentiment=None, df_daily=None) -> dict:
    storage = MongoStorage()
    results = {}
    try:
        if df_news is not None and not df_news.empty:
            results["news"] = storage.upsert_news(df_news.to_dict("records"))
        if df_sentiment is not None and not df_sentiment.empty:
            results["sentiment"] = storage.upsert_sentiment(df_sentiment.to_dict("records"))
        if df_daily is not None and not df_daily.empty:
            results["daily"] = storage.upsert_daily_sentiment(df_daily.to_dict("records"))
    finally:
        storage.close()
    return results