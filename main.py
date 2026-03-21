"""
Entry point for the full Sentiment Pulse pipeline.

Usage:
  python main.py               # run everything end to end
  python main.py --source a    # Source A only
  python main.py --source b    # Source B only
  python main.py --source c    # Source C only
  python main.py --clean-only  # skip scraping, clean existing raw files
  python main.py --store-only  # skip scraping+cleaning, just store to DBs
"""

import asyncio
import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

from scrapers.source_a_playwright import run as run_source_a
from scrapers.source_b_reddit     import run as run_source_b
from scrapers.source_c_yahoo      import run as run_source_c
from analysis.cleaning            import run as run_cleaning
from analysis.sentiment           import run as run_sentiment
from storage.mongo_client         import run as run_mongo
from storage.postgres_client      import run as run_postgres

CLEAN_DIR = Path("data/cleaned")


async def main(source: str = "all", clean_only: bool = False, store_only: bool = False):
    logger.info(f"=== Sentiment Pulse Pipeline | source={source} ===")

    # ── Step 1: Scrape ──────────────────────────────────────────────────────
    if not clean_only and not store_only:
        if source in ("all", "a"):
            logger.info("--- Source A: News RSS ---")
            await run_source_a()

        if source in ("all", "b"):
            logger.info("--- Source B: Alpha Vantage ---")
            run_source_b()

        if source in ("all", "c"):
            logger.info("--- Source C: Yahoo Finance ---")
            run_source_c()

    # ── Step 2: Clean ────────────────────────────────────────────────────────
    if not store_only and source == "all":
        logger.info("--- Cleaning Pipeline ---")
        run_cleaning()

    # ── Step 3: VADER Sentiment ───────────────────────────────────────────────
    if not store_only and source == "all":
        logger.info("--- VADER Sentiment Scoring ---")
        run_sentiment()

    # ── Step 4: Store ─────────────────────────────────────────────────────────
    if source == "all" or store_only:
        logger.info("--- Storage: MongoDB + PostgreSQL ---")

        # Load cleaned parquet files
        def load(name):
            p = CLEAN_DIR / name
            return pd.read_parquet(p) if p.exists() else pd.DataFrame()

        df_news     = load("sentiment_scored_news.parquet")
        df_sentiment = load("source_b_clean.parquet")
        df_daily    = load("daily_sentiment_summary.parquet")
        df_ohlcv    = load("source_c_ohlcv_clean.parquet")
        df_fund     = load("source_c_fund_clean.parquet")

        # MongoDB — unstructured text data
        mongo_results = run_mongo(
            df_news=df_news,
            df_sentiment=df_sentiment,
            df_daily=df_daily,
        )
        logger.success(f"MongoDB: {mongo_results}")

        # PostgreSQL — structured price + sentiment data
        pg_results = run_postgres(
            df_ohlcv=df_ohlcv,
            df_fundamentals=df_fund,
            df_daily_sentiment=df_daily,
        )
        logger.success(f"PostgreSQL: {pg_results}")

    logger.success("=== Pipeline complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",     default="all", choices=["all", "a", "b", "c"])
    parser.add_argument("--clean-only", action="store_true")
    parser.add_argument("--store-only", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.source, args.clean_only, args.store_only))