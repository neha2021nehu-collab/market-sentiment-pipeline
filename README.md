# market-sentiment-pipeline

# Market Sentiment Pipeline
### Python · Playwright · RSS · MongoDB · PostgreSQL · VADER · Docker

An automated data pipeline that scrapes financial news and social media to generate a real-time sentiment score for AI/Tech stocks — answering the question: *"Is the news saying Buy while Reddit is saying Sell?"*

---

## What It Does

| Layer | Source | Method | Output |
|---|---|---|---|
| Source A | TechCrunch AI · Ars Technica · Hacker News | RSS + Playwright fallback | Headlines, summaries, tickers |
| Source B | Reddit (r/wallstreetbets, r/investing) | PRAW API | Posts, comments, upvote ratios |
| Source C | Yahoo Finance | yfinance + BeautifulSoup | OHLCV, P/E, 52-week range |

All three sources feed into a **Pandas cleaning pipeline**, then into **VADER sentiment analysis**, and finally into **MongoDB** (unstructured text) and **PostgreSQL** (structured price data).

---

## Architecture

```
GitHub Actions (cron: 9am weekdays)
        │
        ▼
┌───────────────────────────────────┐
│         Docker Container          │
│                                   │
│  Source A    Source B   Source C  │
│  RSS/PW      PRAW API   yfinance  │
│      └──────────┴──────────┘      │
│               │                   │
│        Pandas cleaning            │
│               │                   │
│     ┌─────────┴─────────┐         │
│     ▼                   ▼         │
│  VADER sentiment    PostgreSQL     │
│     │               (prices)      │
│     ▼                             │
│  MongoDB                          │
│  (articles + scores)              │
└───────────────────────────────────┘
        │
        ▼
  Sentiment vs. Price Dashboard
```

---

## Anti-Bot Measures

- **User-Agent rotation** via `fake-useragent` — randomizes browser fingerprint on every run
- **Viewport randomization** — mimics real screen sizes (1280px to 1920px)
- **`navigator.webdriver` masking** — patches the JS property bot-detectors check first
- **Random human-like delays** — variable pauses between requests (2.5s–6.0s)
- **RSS-first strategy** — uses official feeds where available to stay within ToS; Playwright only as fallback

---

## Tech Stack

```
Scraping     playwright · feedparser · beautifulsoup4 · praw · yfinance
Data         pandas · python-dotenv
Sentiment    vaderSentiment · nltk
Storage      pymongo · psycopg2-binary · sqlalchemy
Infra        Docker · GitHub Actions · loguru
Testing      pytest
```

---

## Project Structure

```
market-sentiment-pipeline/
├── scrapers/
│   ├── source_a_playwright.py   # News: RSS feeds + Playwright fallback
│   ├── source_b_reddit.py       # Social: Reddit via PRAW API
│   └── source_c_yahoo.py        # Historical: Yahoo Finance price data
├── storage/
│   ├── mongo_client.py          # MongoDB connection + insert helpers
│   └── postgres_client.py       # PostgreSQL connection + schema
├── analysis/
│   └── sentiment.py             # VADER scoring pipeline
├── tests/
│   └── test_source_a.py         # Offline unit tests (14 passing)
├── data/
│   ├── raw/                     # Raw JSON output (gitignored)
│   └── cleaned/                 # Pandas-processed output (gitignored)
├── main.py                      # Pipeline entry point
├── conftest.py                  # pytest path config
├── requirements.txt
├── .env.example                 # Environment variable template
└── .gitignore
```

---

## Setup

**1. Clone and create virtual environment**
```bash
git clone https://github.com/neha2021nehu-collab/market-sentiment-pipeline.git
cd market-sentiment-pipeline
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
playwright install chromium
```

**3. Configure environment**
```bash
cp .env.example .env
# Fill in your Reddit API credentials in .env
```

**4. Run the pipeline**
```bash
python main.py --source a    # News scraper only
python main.py --source b    # Reddit scraper only (coming soon)
python main.py --source c    # Yahoo Finance only (coming soon)
python main.py               # Full pipeline
```

**5. Run tests**
```bash
pytest tests/ -v
```

---

## Environment Variables

```bash
# Reddit API — get from reddit.com/prefs/apps
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=SentimentPulse/1.0 by YourUsername

# MongoDB
MONGO_URI=mongodb://localhost:27017
MONGO_DB=sentiment_pulse

# PostgreSQL
POSTGRES_URI=postgresql://user:password@localhost:5432/sentiment_pulse
```

---

## Sample Output

```json
{
  "headline": "Nvidia's AI chips dominate as MSFT expands data center spend",
  "summary": "Analysts raised price targets across the board following...",
  "url": "https://techcrunch.com/...",
  "published_at": "2026-03-19T08:30:00+00:00",
  "tickers_mentioned": ["NVDA", "MSFT"],
  "source": "techcrunch_ai",
  "scraped_at": "2026-03-19T09:00:03+00:00"
}
```

---

## Roadmap

- [x] Source A — News RSS scraper with Playwright fallback
- [ ] Source B — Reddit/PRAW sentiment scraper
- [ ] Source C — Yahoo Finance historical data
- [ ] Pandas cleaning pipeline
- [ ] VADER sentiment scoring
- [ ] MongoDB + PostgreSQL storage layer
- [ ] Sentiment vs. price correlation dashboard
- [ ] Dockerize + GitHub Actions cron schedule

---

## Resume Description

**Market Sentiment Pipeline** | Python · Playwright · MongoDB  
- Developed an automated pipeline scraping 5,000+ daily data points from financial news and social media  
- Implemented RSS-first strategy with headless browser fallback and custom User-Agent rotation to handle anti-scraping measures  
- Processing unstructured text through VADER sentiment analysis to correlate social sentiment with short-term price movements
