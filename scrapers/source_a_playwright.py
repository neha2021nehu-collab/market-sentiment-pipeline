"""
Source A: News scraper for AI/Tech stock sentiment.

Strategy:
  Primary  — RSS feeds from TechCrunch AI + Ars Technica + Hacker News
  Fallback — Playwright headless scrape of TechCrunch AI page

Why these sources over Reuters:
  - Reuters deprecated their public RSS feeds in 2020
  - TechCrunch/Ars Technica have open, maintained feeds — no auth needed
  - Hacker News RSS via hnrss.org is community-curated and very signal-rich
  - All three are ToS-safe for personal/research use

Scraped fields per article:
  - headline, summary, url, published_at, tickers_mentioned, source
"""

import asyncio
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import feedparser
from fake_useragent import UserAgent
from loguru import logger
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RSS_FEEDS = [
    {
        "url":    "https://techcrunch.com/category/artificial-intelligence/feed/",
        "source": "techcrunch_ai",
    },
    {
        "url":    "https://feeds.arstechnica.com/arstechnica/technology-lab",
        "source": "ars_technica",
    },
    {
        "url":    "https://hnrss.org/frontpage?points=100",  # HN posts with 100+ upvotes
        "source": "hacker_news",
    },
    {
        "url":    "https://techcrunch.com/feed/",
        "source": "techcrunch",
    },
]

# Playwright fallback target (if all RSS fail)
PLAYWRIGHT_URL = "https://techcrunch.com/category/artificial-intelligence/"

TRACKED_TICKERS = [
    "NVDA", "MSFT", "GOOGL", "GOOG", "META", "AAPL", "AMZN",
    "AMD", "INTC", "TSLA", "ORCL", "CRM", "PLTR", "AI",
]

# TechCrunch CSS selectors (for Playwright fallback)
SELECTORS = {
    "article_cards": "article.post-block",
    "headline":      "h2.post-block__title a",
    "summary":       "div.post-block__content",
    "timestamp":     "time",
}

RAW_OUTPUT = Path(__file__).parent.parent / "data" / "raw" / "source_a_raw.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_delay(min_s: float = 2.5, max_s: float = 6.0) -> float:
    return random.uniform(min_s, max_s)


def _extract_tickers(text: str) -> list[str]:
    """Find tracked tickers using word-boundary regex — avoids false matches."""
    return [t for t in TRACKED_TICKERS if re.search(rf"\b{t}\b", text)]


def _parse_timestamp(raw: str | None) -> str:
    """Normalize any timestamp string to ISO-8601 UTC."""
    if not raw:
        return datetime.now(timezone.utc).isoformat()
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).isoformat()
    except ValueError:
        return datetime.now(timezone.utc).isoformat()


def _struct_time_to_iso(st) -> str:
    """Convert feedparser's time.struct_time to ISO-8601 UTC string."""
    try:
        return datetime(*st[:6], tzinfo=timezone.utc).isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Strategy 1: RSS Feeds (primary)
# ---------------------------------------------------------------------------

def scrape_rss(max_articles: int = 30) -> list[dict]:
    """
    Parse multiple RSS feeds with feedparser.
    Fast, no browser needed, no bot-detection risk.
    Deduplicates across feeds by URL.
    """
    articles = []
    seen_urls = set()

    for feed_cfg in RSS_FEEDS:
        if len(articles) >= max_articles:
            break

        url    = feed_cfg["url"]
        source = feed_cfg["source"]
        logger.info(f"Fetching RSS [{source}]: {url}")

        feed = feedparser.parse(url)

        if feed.bozo and not feed.entries:
            logger.warning(f"RSS failed for {source}: {getattr(feed, 'bozo_exception', 'unknown error')}")
            continue

        logger.info(f"  → {len(feed.entries)} entries found")

        for entry in feed.entries:
            if len(articles) >= max_articles:
                break

            article_url = entry.get("link", "")
            if article_url in seen_urls:
                continue
            seen_urls.add(article_url)

            headline = entry.get("title", "").strip()
            # feedparser puts summary in 'summary' or 'description'
            summary  = entry.get("summary", entry.get("description", "")).strip()
            # Strip HTML tags that sometimes appear in RSS summaries
            summary  = re.sub(r"<[^>]+>", "", summary).strip()

            published_at = (
                _struct_time_to_iso(entry.published_parsed)
                if entry.get("published_parsed")
                else _parse_timestamp(entry.get("published"))
            )

            full_text = f"{headline} {summary}"

            articles.append({
                "headline":          headline,
                "summary":           summary[:500],  # cap length for storage
                "url":               article_url,
                "published_at":      published_at,
                "tickers_mentioned": _extract_tickers(full_text),
                "source":            source,
                "scraped_at":        datetime.now(timezone.utc).isoformat(),
            })

    logger.success(f"RSS total: {len(articles)} articles")
    return articles


# ---------------------------------------------------------------------------
# Strategy 2: Playwright (fallback)
# ---------------------------------------------------------------------------

class TechCrunchScraper:
    """
    Playwright fallback — scrapes TechCrunch AI section.
    Only runs if all RSS feeds return 0 articles.

    Anti-bot measures:
      - Randomized User-Agent (fake-useragent)
      - Viewport randomization
      - navigator.webdriver masked
      - Human-like random delays between actions
    """

    def __init__(self, headless: bool = True, max_articles: int = 30):
        self.headless    = headless
        self.max_articles = max_articles
        self.ua          = UserAgent()

    async def _build_context(self, playwright):
        browser = await playwright.chromium.launch(
            headless=self.headless,
            args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
        )
        context = await browser.new_context(
            user_agent=self.ua.random,
            viewport={
                "width":  random.choice([1280, 1366, 1440, 1920]),
                "height": random.choice([768, 800, 900, 1080]),
            },
            locale="en-US",
            timezone_id="America/New_York",
        )
        await context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        return browser, context

    async def _extract_article(self, card) -> dict | None:
        try:
            headline_el = await card.query_selector(SELECTORS["headline"])
            if not headline_el:
                return None
            headline = (await headline_el.inner_text()).strip()
            url      = await headline_el.get_attribute("href")

            summary_el = await card.query_selector(SELECTORS["summary"])
            summary    = (await summary_el.inner_text()).strip() if summary_el else ""

            time_el    = await card.query_selector(SELECTORS["timestamp"])
            raw_ts     = await time_el.get_attribute("datetime") if time_el else None

            full_text = f"{headline} {summary}"
            return {
                "headline":          headline,
                "summary":           summary[:500],
                "url":               url,
                "published_at":      _parse_timestamp(raw_ts),
                "tickers_mentioned": _extract_tickers(full_text),
                "source":            "techcrunch_playwright",
                "scraped_at":        datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            logger.warning(f"Card parse failed: {exc}")
            return None

    async def scrape(self) -> list[dict]:
        articles = []
        async with async_playwright() as pw:
            browser, context = await self._build_context(pw)
            page = await context.new_page()
            try:
                logger.info(f"Playwright → {PLAYWRIGHT_URL}")
                await page.goto(PLAYWRIGHT_URL, wait_until="domcontentloaded", timeout=30_000)
                await page.wait_for_selector(SELECTORS["article_cards"], timeout=15_000)
                await asyncio.sleep(_random_delay(1.5, 3.5))

                for _ in range(3):
                    await page.evaluate("window.scrollBy(0, window.innerHeight * 0.8)")
                    await asyncio.sleep(_random_delay(1.0, 2.5))

                cards = await page.query_selector_all(SELECTORS["article_cards"])
                logger.info(f"Found {len(cards)} article cards")

                for card in cards[: self.max_articles]:
                    article = await self._extract_article(card)
                    if article:
                        articles.append(article)
                    await asyncio.sleep(_random_delay(0.3, 0.9))

            except PWTimeout:
                logger.error("Playwright timed out")
            except Exception as exc:
                logger.error(f"Playwright error: {exc}")
            finally:
                await browser.close()

        logger.success(f"Playwright scraped {len(articles)} articles")
        return articles


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

async def run(max_articles: int = 30) -> list[dict]:
    """
    Run Source A:
      1. Try RSS feeds (fast, reliable, ToS-safe)
      2. Fall back to Playwright if RSS yields nothing
    """
    articles = scrape_rss(max_articles=max_articles)

    if not articles:
        logger.warning("RSS returned 0 — falling back to Playwright")
        scraper  = TechCrunchScraper(headless=True, max_articles=max_articles)
        articles = await scraper.scrape()

    save_raw(articles)
    return articles


def save_raw(articles: list[dict], path: Path = RAW_OUTPUT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(articles)} articles → {path}")


if __name__ == "__main__":
    asyncio.run(run())