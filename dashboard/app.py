"""
Flask server for the Sentiment Pulse dashboard.
Serves the HTML dashboard and provides API endpoints
that read from cleaned Parquet files.

Run from project root: python dashboard/app.py
Open: http://localhost:5000
"""

import json
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, send_from_directory
from loguru import logger

app = Flask(__name__, static_folder="static")

# Always resolve paths relative to project root (parent of dashboard/)
PROJECT_ROOT = Path(__file__).parent.parent
CLEAN_DIR    = PROJECT_ROOT / "data" / "cleaned"

TICKERS = ["NVDA", "MSFT", "GOOGL", "META", "AAPL",
           "AMZN", "AMD", "TSLA", "PLTR", "ORCL"]


def load(name: str) -> pd.DataFrame:
    p = CLEAN_DIR / name
    if not p.exists():
        logger.warning(f"File not found: {p}")
        return pd.DataFrame()
    return pd.read_parquet(p)


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.route("/api/overview")
def overview():
    df_b    = load("source_b_clean.parquet")
    df_fund = load("source_c_fund_clean.parquet")
    df_news = load("sentiment_scored_news.parquet")

    bullish = bearish = neutral = 0
    if not df_b.empty:
        counts  = df_b["ticker_sentiment_label"].value_counts().to_dict()
        bullish = int(counts.get("bullish", 0))
        bearish = int(counts.get("bearish", 0))
        neutral = int(counts.get("neutral", 0))

    avg_pe = None
    if not df_fund.empty and "pe_ratio" in df_fund.columns:
        avg_pe = round(float(df_fund["pe_ratio"].mean(skipna=True)), 1)

    avg_vader = None
    if not df_news.empty and "vader_compound" in df_news.columns:
        avg_vader = round(float(df_news["vader_compound"].mean()), 4)

    return jsonify({
        "total_articles": bullish + bearish + neutral,
        "bullish":        bullish,
        "bearish":        bearish,
        "neutral":        neutral,
        "avg_pe":         avg_pe,
        "avg_vader":      avg_vader,
        "tickers":        TICKERS,
    })


@app.route("/api/sentiment/<ticker>")
def sentiment(ticker: str):
    ticker = ticker.upper()
    df     = load("source_b_clean.parquet")

    if df.empty:
        return jsonify({"error": "No data"}), 404

    df_t   = df[df["ticker"] == ticker]
    counts = df_t["ticker_sentiment_label"].value_counts().to_dict()

    recent = (
        df_t.sort_values("published_at", ascending=False)
        .head(5)[["title", "ticker_sentiment_label", "ticker_sentiment_score", "published_at"]]
        .copy()
    )
    recent["published_at"] = recent["published_at"].astype(str)

    return jsonify({
        "ticker":   ticker,
        "bullish":  int(counts.get("bullish", 0)),
        "bearish":  int(counts.get("bearish", 0)),
        "neutral":  int(counts.get("neutral", 0)),
        "articles": recent.to_dict("records"),
    })


@app.route("/api/price/<ticker>")
def price(ticker: str):
    ticker = ticker.upper()
    df     = load("source_c_ohlcv_clean.parquet")

    if df.empty:
        return jsonify({"error": "No data"}), 404

    df_t = (
        df[df["ticker"] == ticker]
        .sort_values("date")
        .tail(30)
        .copy()
    )
    df_t["date"] = df_t["date"].astype(str)

    return jsonify({
        "ticker": ticker,
        "dates":  df_t["date"].tolist(),
        "close":  df_t["close"].round(2).tolist(),
        "ma_5":   df_t["ma_5"].round(2).tolist(),
        "ma_20":  df_t["ma_20"].round(2).tolist(),
        "volume": df_t["volume"].tolist(),
    })


@app.route("/api/fundamentals")
def fundamentals():
    df = load("source_c_fund_clean.parquet")
    if df.empty:
        return jsonify([])

    cols = ["ticker", "company_name", "pe_ratio", "market_cap",
            "fifty_two_week_high", "fifty_two_week_low", "analyst_target_price"]
    cols = [c for c in cols if c in df.columns]
    return jsonify(df[cols].to_dict("records"))


@app.route("/api/news")
def news():
    df = load("sentiment_scored_news.parquet")
    if df.empty:
        return jsonify([])

    cols = ["headline", "vader_label", "vader_compound",
            "tickers_mentioned", "source", "published_at", "url"]
    cols = [c for c in cols if c in df.columns]

    recent = df.sort_values("published_at", ascending=False).head(20)[cols].copy()
    recent["published_at"] = recent["published_at"].astype(str)
    recent["tickers_mentioned"] = recent["tickers_mentioned"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    return jsonify(recent.to_dict("records"))


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


if __name__ == "__main__":
    logger.info(f"Data directory: {CLEAN_DIR}")
    logger.info(f"Files found: {list(CLEAN_DIR.glob('*.parquet')) if CLEAN_DIR.exists() else 'DIRECTORY NOT FOUND'}")
    logger.info("Dashboard running → http://localhost:5000")
    app.run(debug=True, port=5000)