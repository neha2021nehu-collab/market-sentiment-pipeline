"""
VADER Sentiment Scoring Pipeline.

What this does:
  - Loads cleaned Source A news articles (Parquet)
  - Runs VADER sentiment analysis on headline + summary
  - Produces compound score (-1.0 to +1.0) per article
  - Aggregates daily sentiment score per ticker
  - Merges with Source B Alpha Vantage scores for comparison
  - Saves final combined sentiment report to data/cleaned/

Why VADER:
  - Trained specifically on social/news text (not books)
  - Handles punctuation emphasis ("NVDA SURGES!!!" scores higher than "nvda surges")
  - No model training needed — rule-based, deterministic
  - Compound score is directly interpretable:
      >= 0.05  → positive/bullish
      <= -0.05 → negative/bearish
      between  → neutral

Output files:
  - sentiment_scored_news.parquet   — article-level VADER scores
  - daily_sentiment_summary.parquet — daily aggregated score per ticker
  - sentiment_comparison.parquet    — VADER vs Alpha Vantage side by side
"""

import json
from pathlib import Path
from datetime import timezone

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from loguru import logger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CLEAN_DIR = Path(__file__).parent.parent / "data" / "cleaned"
OUTPUT_DIR = CLEAN_DIR  # save results alongside other cleaned files

# ---------------------------------------------------------------------------
# VADER thresholds (standard convention)
# ---------------------------------------------------------------------------

BULLISH_THRESHOLD =  0.05
BEARISH_THRESHOLD = -0.05


def compound_to_label(score: float) -> str:
    """Convert VADER compound score to bullish/neutral/bearish label."""
    if score >= BULLISH_THRESHOLD:
        return "bullish"
    elif score <= BEARISH_THRESHOLD:
        return "bearish"
    return "neutral"


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

class VADERScorer:
    """
    Runs VADER sentiment analysis on news article text.

    Scores both headline and summary separately, then combines them
    with headline weighted more heavily (news headlines are written
    to be punchy and signal-rich).
    """

    HEADLINE_WEIGHT = 0.6
    SUMMARY_WEIGHT  = 0.4

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        logger.info("VADER SentimentIntensityAnalyzer initialized")

    def score_text(self, text: str) -> dict:
        """Return full VADER scores for a single text string."""
        if not text or not text.strip():
            return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
        return self.analyzer.polarity_scores(text)

    def score_article(self, headline: str, summary: str) -> dict:
        """
        Score an article using weighted combination of headline + summary.
        Headline carries more weight — it's the most signal-dense part.
        """
        h_scores = self.score_text(headline)
        s_scores = self.score_text(summary)

        combined_compound = (
            h_scores["compound"] * self.HEADLINE_WEIGHT +
            s_scores["compound"] * self.SUMMARY_WEIGHT
        )

        return {
            "vader_headline_compound": round(h_scores["compound"], 4),
            "vader_summary_compound":  round(s_scores["compound"], 4),
            "vader_compound":          round(combined_compound, 4),
            "vader_label":             compound_to_label(combined_compound),
            "vader_positive":          round(h_scores["pos"], 4),
            "vader_negative":          round(h_scores["neg"], 4),
            "vader_neutral":           round(h_scores["neu"], 4),
        }

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply VADER scoring to every row in a news DataFrame."""
        logger.info(f"Scoring {len(df)} articles with VADER...")

        scores = df.apply(
            lambda row: self.score_article(
                row.get("headline", ""),
                row.get("summary", ""),
            ),
            axis=1,
            result_type="expand",
        )

        df_scored = pd.concat([df, scores], axis=1)
        logger.success(f"VADER scoring complete: {len(df_scored)} articles scored")
        return df_scored


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def build_daily_sentiment(df_scored: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate article-level VADER scores into a daily sentiment score
    per ticker.

    For tickers_mentioned we explode the list so each ticker gets
    its own row, then group by ticker + date.
    """
    logger.info("Building daily sentiment aggregation...")

    # Explode tickers_mentioned so NVDA article appears in NVDA's group
    df = df_scored.copy()
    df["date"] = pd.to_datetime(df["published_at"]).dt.date

    # Handle tickers_mentioned being a list column
    df = df.explode("tickers_mentioned").rename(
        columns={"tickers_mentioned": "ticker"}
    )

    # Drop rows where no ticker was found
    df = df[df["ticker"].notna() & (df["ticker"] != "")]

    if df.empty:
        logger.warning("No ticker-tagged articles found for daily aggregation")
        return pd.DataFrame()

    daily = (
        df.groupby(["ticker", "date"])
        .agg(
            article_count    = ("vader_compound", "count"),
            avg_vader_score  = ("vader_compound", "mean"),
            max_vader_score  = ("vader_compound", "max"),
            min_vader_score  = ("vader_compound", "min"),
            bullish_count    = ("vader_label", lambda x: (x == "bullish").sum()),
            bearish_count    = ("vader_label", lambda x: (x == "bearish").sum()),
            neutral_count    = ("vader_label", lambda x: (x == "neutral").sum()),
        )
        .reset_index()
    )

    daily["avg_vader_score"] = daily["avg_vader_score"].round(4)
    daily["dominant_label"]  = daily["avg_vader_score"].apply(compound_to_label)

    logger.success(f"Daily sentiment: {len(daily)} ticker-day rows")
    return daily


def build_sentiment_comparison(
    df_scored: pd.DataFrame,
    df_source_b: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare VADER-scored news (Source A) with Alpha Vantage
    pre-labeled sentiment (Source B) on a per-ticker daily basis.

    This is the core insight of the project:
    'Is VADER saying Bullish while Alpha Vantage says Bearish?'
    """
    logger.info("Building VADER vs Alpha Vantage comparison...")

    # Daily VADER average per ticker
    df_a = df_scored.copy()
    df_a["date"] = pd.to_datetime(df_a["published_at"]).dt.date
    df_a = df_a.explode("tickers_mentioned").rename(
        columns={"tickers_mentioned": "ticker"}
    )
    df_a = df_a[df_a["ticker"].notna() & (df_a["ticker"] != "")]

    vader_daily = (
        df_a.groupby(["ticker", "date"])
        .agg(vader_avg=("vader_compound", "mean"))
        .reset_index()
    )
    vader_daily["vader_label"] = vader_daily["vader_avg"].apply(compound_to_label)

    # Daily Alpha Vantage average per ticker
    df_b = df_source_b.copy()
    df_b["date"] = pd.to_datetime(df_b["published_at"]).dt.date

    av_daily = (
        df_b.groupby(["ticker", "date"])
        .agg(av_avg=("combined_sentiment_score", "mean"))
        .reset_index()
    )
    av_daily["av_label"] = av_daily["av_avg"].apply(compound_to_label)

    # Merge on ticker + date
    comparison = pd.merge(vader_daily, av_daily, on=["ticker", "date"], how="outer")

    # Flag disagreements — where VADER and AV diverge
    comparison["signals_agree"] = comparison["vader_label"] == comparison["av_label"]
    comparison["signal_conflict"] = (
        (comparison["vader_label"] == "bullish") & (comparison["av_label"] == "bearish") |
        (comparison["vader_label"] == "bearish") & (comparison["av_label"] == "bullish")
    )

    logger.success(
        f"Comparison built: {len(comparison)} rows | "
        f"Conflicts: {comparison['signal_conflict'].sum()}"
    )
    return comparison


# ---------------------------------------------------------------------------
# Save + Run
# ---------------------------------------------------------------------------

def save_results(
    df_scored:    pd.DataFrame,
    df_daily:     pd.DataFrame,
    df_comparison: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "sentiment_scored_news.parquet":   df_scored,
        "daily_sentiment_summary.parquet": df_daily,
        "sentiment_comparison.parquet":    df_comparison,
    }
    for filename, df in outputs.items():
        if df.empty:
            logger.warning(f"Skipping empty: {filename}")
            continue
        path = OUTPUT_DIR / filename
        df.to_parquet(path, index=False)
        logger.info(f"Saved {len(df)} rows → {path}")


def print_insights(df_scored, df_daily, df_comparison) -> None:
    """Print the key business insights from the sentiment analysis."""
    logger.info("\n" + "="*55)
    logger.info("SENTIMENT ANALYSIS INSIGHTS")
    logger.info("="*55)

    if not df_scored.empty:
        label_dist = df_scored["vader_label"].value_counts().to_dict()
        avg_score  = df_scored["vader_compound"].mean()
        logger.info(f"VADER News Sentiment: {label_dist}")
        logger.info(f"Average VADER compound score: {avg_score:.4f}")

    if not df_daily.empty:
        top_bullish = df_daily.nlargest(3, "avg_vader_score")[["ticker", "date", "avg_vader_score"]]
        logger.info(f"Most bullish ticker-days:\n{top_bullish.to_string(index=False)}")

    if not df_comparison.empty:
        conflicts = df_comparison[df_comparison["signal_conflict"] == True]
        logger.info(f"Signal conflicts (VADER vs AV): {len(conflicts)} ticker-days")
        if not conflicts.empty:
            logger.info(f"  Example conflict:\n{conflicts[['ticker','date','vader_label','av_label']].head(3).to_string(index=False)}")


def run() -> dict:
    """Run full VADER sentiment pipeline."""
    logger.info("=== Starting VADER Sentiment Pipeline ===")

    # Load cleaned Source A
    source_a_path = CLEAN_DIR / "source_a_clean.parquet"
    source_b_path = CLEAN_DIR / "source_b_clean.parquet"

    if not source_a_path.exists():
        logger.error(f"Source A cleaned file not found: {source_a_path}")
        logger.error("Run 'python analysis/cleaning.py' first")
        return {}

    df_news      = pd.read_parquet(source_a_path)
    df_source_b  = pd.read_parquet(source_b_path) if source_b_path.exists() else pd.DataFrame()

    # Score with VADER
    scorer    = VADERScorer()
    df_scored = scorer.score_dataframe(df_news)

    # Aggregate + compare
    df_daily      = build_daily_sentiment(df_scored)
    df_comparison = build_sentiment_comparison(df_scored, df_source_b) if not df_source_b.empty else pd.DataFrame()

    # Print insights + save
    print_insights(df_scored, df_daily, df_comparison)
    save_results(df_scored, df_daily, df_comparison)

    logger.success("=== VADER pipeline complete ===")
    return {
        "scored_news":  df_scored,
        "daily":        df_daily,
        "comparison":   df_comparison,
    }


if __name__ == "__main__":
    run()