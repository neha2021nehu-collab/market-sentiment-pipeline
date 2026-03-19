"""
Entry point for the full pipeline.
Currently wires Source A only — Sources B and C will be added in next sessions.

Usage:
  python main.py            # run full pipeline
  python main.py --source a # run only Source A
"""

import asyncio
import argparse
from loguru import logger
from scrapers.source_a_playwright import run as run_source_a


async def main(source: str = "all"):
    logger.info(f"=== Sentiment Pulse Pipeline | source={source} ===")

    if source in ("all", "a"):
        logger.info("--- Source A: Reuters Technology ---")
        articles = await run_source_a()
        logger.success(f"Source A complete: {len(articles)} articles")

    # Source B and C placeholders — coming next
    if source in ("all", "b"):
        logger.warning("Source B (Reddit) not yet implemented")

    if source in ("all", "c"):
        logger.warning("Source C (Yahoo Finance) not yet implemented")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="all", choices=["all", "a", "b", "c"])
    args = parser.parse_args()
    asyncio.run(main(args.source))