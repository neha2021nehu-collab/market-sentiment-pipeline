"""
Entry point for the full Sentiment Pulse pipeline.

Usage:
  python main.py            # run all sources
  python main.py --source a # Source A only (news RSS)
  python main.py --source b # Source B only (StockTwits social)
  python main.py --source c # Source C only (Yahoo Finance) — coming soon
"""

import asyncio
import argparse
from loguru import logger
from scrapers.source_a_playwright import run as run_source_a
from scrapers.source_b_reddit import run as run_source_b


async def main(source: str = "all"):
    logger.info(f"=== Sentiment Pulse Pipeline | source={source} ===")

    if source in ("all", "a"):
        logger.info("--- Source A: News RSS + Playwright ---")
        articles = await run_source_a()
        logger.success(f"Source A complete: {len(articles)} articles")

    if source in ("all", "b"):
        logger.info("--- Source B: StockTwits Social Sentiment ---")
        posts = run_source_b()
        logger.success(f"Source B complete: {len(posts)} posts")

    if source in ("all", "c"):
        logger.warning("Source C (Yahoo Finance) not yet implemented")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="all", choices=["all", "a", "b", "c"])
    args = parser.parse_args()
    asyncio.run(main(args.source))