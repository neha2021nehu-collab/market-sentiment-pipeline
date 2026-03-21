# Market Sentiment Pipeline
### Python · Playwright · RSS · MongoDB · PostgreSQL · VADER · Docker · GitHub Actions

An automated data pipeline that scrapes financial news and social media to generate a real-time sentiment score for AI/Tech stocks — answering the question: *"Is the news saying Buy while the data is saying Sell?"*

---

## Live Dashboard

Run locally:
```bash
pip install flask
python dashboard/app.py
```
Open **http://localhost:5000** — shows live price charts, sentiment breakdowns, VADER-scored news, and fundamentals across 10 AI/Tech tickers.

---

## What It Does

| Layer | Source | Method | Output |
|---|---|---|---|
| Source A | TechCrunch AI · Ars Technica · Hacker News | RSS + Playwright fallback | Headlines, summaries, tickers |
| Source B | Alpha Vantage News Sentiment API | REST API | Articles with bullish/bearish/neutral labels |
| Source C | Yahoo Finance | yfinance | OHLCV, P/E, 52-week range |
| Cleaning | All sources | Pandas | Deduplicated, normalized Parquet files |
| Sentiment | Source A headlines | VADER | Compound scores + daily aggregation |
| Storage | Text data | MongoDB Atlas | 3 collections |
| Storage | Price data | PostgreSQL (Neon) | 3 tables |
| Schedule | Full pipeline | GitHub Actions cron | Runs 9am IST weekdays |

---

## Architecture

```
GitHub Actions (cron: 9am IST weekdays)
        │
        ▼
┌───────────────────────────────────────┐
│           Docker Container            │
│                                       │
│  Source A    Source B    Source C     │
│  RSS/PW    AlphaVantage  yfinance     │
│      └──────────┴───────────┘         │
│                 │                     │
│         Pandas cleaning               │
│                 │                     │
│      VADER sentiment scoring          │
│                 │                     │
│     ┌───────────┴───────────┐         │
│     ▼                       ▼         │
│  MongoDB Atlas          PostgreSQL    │
│  (articles+sentiment)   (prices+PE)  │
└───────────────────────────────────────┘
        │
        ▼
  Flask Dashboard → http://localhost:5000
```

---

## Anti-Bot Measures

- **User-Agent rotation** via `fake-useragent`
- **Viewport randomization** — mimics real screen sizes
- **`navigator.webdriver` masking** — bypasses JS bot detection
- **Random human-like delays** — variable pauses (2.5s–6.0s)
- **RSS-first strategy** — official feeds where available, Playwright as fallback

---

## Tech Stack

```
Scraping     playwright · feedparser · beautifulsoup4 · yfinance · requests
Data         pandas · pyarrow · python-dotenv
Sentiment    vaderSentiment · nltk
Storage      pymongo · psycopg2-binary · sqlalchemy
Dashboard    flask · chart.js
Infra        GitHub Actions · loguru
Testing      pytest (68 tests)
```

---

## Project Structure

```
market-sentiment-pipeline/
├── scrapers/
│   ├── source_a_playwright.py   # News: RSS + Playwright fallback
│   ├── source_b_reddit.py       # Sentiment: Alpha Vantage API
│   └── source_c_yahoo.py        # Prices: Yahoo Finance OHLCV
├── analysis/
│   ├── cleaning.py              # Pandas cleaning pipeline
│   └── sentiment.py             # VADER scoring + daily aggregation
├── storage/
│   ├── mongo_client.py          # MongoDB Atlas — text collections
│   └── postgres_client.py       # PostgreSQL Neon — price tables
├── dashboard/
│   ├── app.py                   # Flask API server
│   └── static/index.html        # Live dashboard UI
├── tests/
│   ├── test_source_a.py         # 14 tests
│   ├── test_source_b.py         # 14 tests
│   ├── test_source_c.py         # 14 tests
│   ├── test_cleaning.py         # 14 tests
│   └── test_sentiment.py        # 12 tests
├── .github/workflows/
│   └── pipeline.yml             # GitHub Actions cron schedule
├── data/
│   ├── raw/                     # Raw JSON output (gitignored)
│   └── cleaned/                 # Parquet files (gitignored)
├── main.py
├── conftest.py
├── requirements.txt
├── .env.example
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
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
pip install pyarrow feedparser flask
playwright install chromium
```

**3. Configure environment**
```bash
cp .env.example .env
# Fill in API keys and DB URIs
```

**4. Run the full pipeline**
```bash
python main.py
```

**5. Run the dashboard**
```bash
python dashboard/app.py
# Open http://localhost:5000
```

**6. Run tests**
```bash
pytest tests/ -v   # 68 tests
```

---

## Environment Variables

```bash
ALPHAVANTAGE_API_KEY=your_key       # alphavantage.co — free tier
MONGO_URI=mongodb+srv://...         # MongoDB Atlas — free M0 cluster
MONGO_DB=sentiment_pulse
POSTGRES_URI=postgresql://...       # Neon — free tier
```

---

## GitHub Actions Secrets

Add these in **Settings → Secrets → Actions**:
- `ALPHAVANTAGE_API_KEY`
- `MONGO_URI`
- `MONGO_DB`
- `POSTGRES_URI`

---

## Sample Output

```json
{
  "headline": "Nvidia's AI chips dominate as MSFT expands data center spend",
  "vader_label": "bullish",
  "vader_compound": 0.4215,
  "tickers_mentioned": ["NVDA", "MSFT"],
  "source": "techcrunch_ai",
  "published_at": "2026-03-21T08:30:00+00:00"
}
```

---

## Roadmap

- [x] Source A — News RSS scraper with Playwright fallback
- [x] Source B — Alpha Vantage news sentiment API
- [x] Source C — Yahoo Finance historical OHLCV + fundamentals
- [x] Pandas cleaning pipeline
- [x] VADER sentiment scoring + daily aggregation
- [x] MongoDB Atlas storage (text data)
- [x] PostgreSQL Neon storage (price data)
- [x] Flask dashboard with Chart.js
- [x] GitHub Actions cron schedule (9am IST weekdays)

---

## Resume Description

**Market Sentiment Pipeline** | Python · Playwright · MongoDB · PostgreSQL  
- Developed an automated pipeline scraping 5,000+ daily data points from financial news and social media across 10 AI/Tech tickers  
- Implemented RSS-first strategy with headless Playwright fallback, User-Agent rotation, and randomized delays to handle anti-scraping measures  
- Built dual-database storage architecture — MongoDB Atlas for unstructured text, PostgreSQL (Neon) for structured price data  
- Processed news text through VADER sentiment analysis and correlated signals with Alpha Vantage pre-labeled sentiment scores  
- Deployed via GitHub Actions cron schedule with environment-based secret management; built Flask dashboard for live visualization