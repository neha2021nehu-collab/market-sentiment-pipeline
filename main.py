"""
Entry point for the full Sentiment Pulse pipeline.

Usage:
  python main.py              # run all sources + cleaning
  python main.py --source a   # Source A only
  python main.py --source b   # Source B only
  python main.py --source c   # Source C only
  python main.py --clean-only # skip scraping, just clean existing raw files
"""

import asyncio
import argparse
from loguru import logger
from scrapers.source_a_playwright import run as run_source_a
from scrapers.source_b_reddit import run as run_source_b
from scrapers.source_c_yahoo import run as run_source_c
from analysis.cleaning import run as run_cleaning


async def main(source: str = "all", clean_only: bool = False):
    logger.info(f"=== Sentiment Pulse Pipeline | source={source} ===")

    if not clean_only:
        if source in ("all", "a"):
            logger.info("--- Source A: News RSS + Playwright ---")
            articles = await run_source_a()
            logger.success(f"Source A: {len(articles)} articles")

        if source in ("all", "b"):
            logger.info("--- Source B: Alpha Vantage News Sentiment ---")
            posts = run_source_b()
            logger.success(f"Source B: {len(posts)} articles")

        if source in ("all", "c"):
            logger.info("--- Source C: Yahoo Finance Price Data ---")
            ohlcv, fundamentals = run_source_c()
            logger.success(f"Source C: {len(ohlcv)} OHLCV | {len(fundamentals)} fundamentals")

    if source == "all" or clean_only:
        logger.info("--- Cleaning Pipeline ---")
        dfs = run_cleaning()
        logger.success(f"Cleaning complete: {sum(len(v) for v in dfs.values())} total rows across all datasets")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="all", choices=["all", "a", "b", "c"])
    parser.add_argument("--clean-only", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.source, args.clean_only))