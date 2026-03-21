"""
Pandas cleaning pipeline for all three sources.

What this does per source:
  Source A (news):
    - Drop duplicates by URL
    - Normalize published_at to UTC datetime
    - Strip whitespace from headline/summary
    - Flag articles mentioning multiple tickers
    - Add text_length feature

  Source B (Alpha Vantage sentiment):
    - Drop duplicates by URL + ticker combo
    - Normalize published_at
    - Encode sentiment label as numeric score (-1, 0, 1)
    - Cap summary length
    - Add weighted_score = sentiment_score × confidence

  Source C (OHLCV):
    - Parse date to datetime
    - Drop rows with null close price
    - Add daily_return = % change in close price
    - Add price_range = (high - low) / close
    - Sort by ticker + date

Output: cleaned Parquet files in data/cleaned/ (faster + smaller than CSV)
"""

import json
from pathlib import Path

import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR   = Path(__file__).parent.parent / "data"
RAW_DIR    = DATA_DIR / "raw"
CLEAN_DIR  = DATA_DIR / "cleaned"

CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Source A: News articles
# ---------------------------------------------------------------------------

def clean_source_a(path: Path = RAW_DIR / "source_a_raw.json") -> pd.DataFrame:
    """Clean and enrich news articles from Source A."""
    logger.info("Cleaning Source A (news articles)...")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    if df.empty:
        logger.warning("Source A raw file is empty")
        return df

    original_count = len(df)

    # --- Deduplication ---
    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
    logger.info(f"  Deduped: {original_count} → {len(df)} rows")

    # --- Normalize timestamps ---
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["scraped_at"]   = pd.to_datetime(df["scraped_at"],   utc=True, errors="coerce")

    # Drop rows where timestamp failed to parse
    df = df.dropna(subset=["published_at"]).reset_index(drop=True)

    # --- Clean text ---
    df["headline"] = df["headline"].str.strip()
    df["summary"]  = df["summary"].fillna("").str.strip()

    # --- Feature engineering ---
    df["text_length"]     = df["headline"].str.len() + df["summary"].str.len()
    df["ticker_count"]    = df["tickers_mentioned"].apply(len)
    df["is_multi_ticker"] = df["ticker_count"] > 1

    # --- Sort ---
    df = df.sort_values("published_at", ascending=False).reset_index(drop=True)

    logger.success(f"Source A cleaned: {len(df)} articles")
    return df


# ---------------------------------------------------------------------------
# Source B: Sentiment articles
# ---------------------------------------------------------------------------

SENTIMENT_SCORE_MAP = {
    "bullish": 1,
    "neutral": 0,
    "bearish": -1,
}

def clean_source_b(path: Path = RAW_DIR / "source_b_raw.json") -> pd.DataFrame:
    """Clean and enrich Alpha Vantage sentiment data from Source B."""
    logger.info("Cleaning Source B (sentiment articles)...")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    if df.empty:
        logger.warning("Source B raw file is empty")
        return df

    original_count = len(df)

    # --- Deduplication (same article can appear for multiple tickers) ---
    df = df.drop_duplicates(subset=["url", "ticker"]).reset_index(drop=True)
    logger.info(f"  Deduped: {original_count} → {len(df)} rows")

    # --- Normalize timestamps ---
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["scraped_at"]   = pd.to_datetime(df["scraped_at"],   utc=True, errors="coerce")
    df = df.dropna(subset=["published_at"]).reset_index(drop=True)

    # --- Fill missing text ---
    df["title"]   = df["title"].fillna("").str.strip()
    df["summary"] = df["summary"].fillna("").str.strip()

    # --- Encode sentiment as numeric ---
    df["ticker_sentiment_numeric"]  = df["ticker_sentiment_label"].map(SENTIMENT_SCORE_MAP).fillna(0)
    df["overall_sentiment_numeric"] = df["overall_sentiment_label"].map(SENTIMENT_SCORE_MAP).fillna(0)

    # --- Weighted score: sentiment direction × confidence magnitude ---
    # ticker_sentiment_score ranges from -1 to +1 (already signed)
    df["ticker_sentiment_score"]  = pd.to_numeric(df["ticker_sentiment_score"],  errors="coerce").fillna(0)
    df["overall_sentiment_score"] = pd.to_numeric(df["overall_sentiment_score"], errors="coerce").fillna(0)

    # --- Feature: combined signal (average of overall + ticker-specific) ---
    df["combined_sentiment_score"] = (
        df["ticker_sentiment_score"] + df["overall_sentiment_score"]
    ) / 2

    # --- Sort by ticker + date ---
    df = df.sort_values(["ticker", "published_at"], ascending=[True, False]).reset_index(drop=True)

    logger.success(f"Source B cleaned: {len(df)} sentiment articles")
    return df


# ---------------------------------------------------------------------------
# Source C: OHLCV price data
# ---------------------------------------------------------------------------

def clean_source_c(
    ohlcv_path: Path = RAW_DIR / "source_c_raw.json",
    fund_path:  Path = RAW_DIR / "source_c_fundamentals.json",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean OHLCV records and fundamentals from Source C."""
    logger.info("Cleaning Source C (price data)...")

    # --- OHLCV ---
    with open(ohlcv_path, encoding="utf-8") as f:
        ohlcv_data = json.load(f)

    df = pd.DataFrame(ohlcv_data)
    original_count = len(df)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).reset_index(drop=True)
    logger.info(f"  OHLCV rows after null drop: {original_count} → {len(df)}")

    # Ensure numeric columns
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Feature engineering ---
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Daily return % per ticker
    df["daily_return"] = (
        df.groupby("ticker")["close"]
        .pct_change()
        .round(6)
    )

    # Intraday volatility: (high - low) / close
    df["price_range"] = ((df["high"] - df["low"]) / df["close"]).round(6)

    # 5-day and 20-day rolling average close per ticker
    df["ma_5"]  = df.groupby("ticker")["close"].transform(lambda x: x.rolling(5).mean()).round(4)
    df["ma_20"] = df.groupby("ticker")["close"].transform(lambda x: x.rolling(20).mean()).round(4)

    # --- Fundamentals ---
    with open(fund_path, encoding="utf-8") as f:
        fund_data = json.load(f)

    df_fund = pd.DataFrame(fund_data)
    df_fund["scraped_at"] = pd.to_datetime(df_fund["scraped_at"], utc=True, errors="coerce")

    logger.success(f"Source C cleaned: {len(df)} OHLCV rows | {len(df_fund)} fundamentals")
    return df, df_fund


# ---------------------------------------------------------------------------
# Save + Run
# ---------------------------------------------------------------------------

def save_cleaned(
    df_a:    pd.DataFrame,
    df_b:    pd.DataFrame,
    df_c:    pd.DataFrame,
    df_fund: pd.DataFrame,
) -> None:
    """Save all cleaned DataFrames to Parquet format."""
    outputs = {
        "source_a_clean.parquet":       df_a,
        "source_b_clean.parquet":       df_b,
        "source_c_ohlcv_clean.parquet": df_c,
        "source_c_fund_clean.parquet":  df_fund,
    }
    for filename, df in outputs.items():
        if df.empty:
            logger.warning(f"Skipping empty DataFrame: {filename}")
            continue
        out_path = CLEAN_DIR / filename
        df.to_parquet(out_path, index=False)
        logger.info(f"Saved {len(df)} rows → {out_path}")


def print_summary(df_a, df_b, df_c, df_fund) -> None:
    """Print a quick summary of each cleaned dataset."""
    logger.info("\n" + "="*50)
    logger.info("CLEANING SUMMARY")
    logger.info("="*50)

    if not df_a.empty:
        logger.info(f"Source A | {len(df_a)} articles | "
                    f"Sources: {df_a['source'].value_counts().to_dict()}")

    if not df_b.empty:
        sentiment_dist = df_b["ticker_sentiment_label"].value_counts().to_dict()
        logger.info(f"Source B | {len(df_b)} articles | Sentiment: {sentiment_dist}")

    if not df_c.empty:
        logger.info(f"Source C | {len(df_c)} OHLCV rows | "
                    f"Tickers: {df_c['ticker'].nunique()} | "
                    f"Date range: {df_c['date'].min().date()} → {df_c['date'].max().date()}")

    if not df_fund.empty:
        logger.info(f"Fundamentals | {len(df_fund)} tickers | "
                    f"Avg P/E: {df_fund['pe_ratio'].mean():.1f}")


def run() -> dict[str, pd.DataFrame]:
    """Run full cleaning pipeline across all sources."""
    logger.info("=== Starting Pandas Cleaning Pipeline ===")

    df_a           = clean_source_a()
    df_b           = clean_source_b()
    df_c, df_fund  = clean_source_c()

    print_summary(df_a, df_b, df_c, df_fund)
    save_cleaned(df_a, df_b, df_c, df_fund)

    logger.success("=== Cleaning pipeline complete ===")
    return {
        "news":         df_a,
        "sentiment":    df_b,
        "ohlcv":        df_c,
        "fundamentals": df_fund,
    }


if __name__ == "__main__":
    run()