"""Sentiment analysis pipeline for innovation showcase project.

This module automates Step 1 of Phase 3:
1. Collect corruption-related news headlines per country/year.
2. Score sentiment with VADER/TextBlob and aggregate yearly averages.
3. Merge the sentiment feature into the main corruption dataset.

Usage (from project root):
    python -m src.sentiment_analysis --provider newsapi --start-year 2010 --end-year 2023

Environment:
    NEWSAPI_KEY must be set (or passed via --newsapi-key) when using the NewsAPI provider.
"""

from __future__ import annotations

import argparse
import calendar
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SENTIMENT_DIR = DATA_DIR / "sentiment"
RAW_NEWS_FILE = SENTIMENT_DIR / "news_headlines_raw.csv"
SENTIMENT_SCORES_FILE = SENTIMENT_DIR / "sentiment_scores.csv"
BASE_DATASET_FILE = DATA_DIR / "processed" / "corruption_data_expanded_labeled.csv"
FINAL_TRAINING_FILE = DATA_DIR / "processed" / "final_training_data.csv"

DEFAULT_KEYWORDS = ["corruption", "bribery", '"money laundering"', "fraud"]


def configure_logging(level: str) -> None:
    """Set up project-wide logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_directories() -> None:
    """Ensure output directories exist."""
    SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sentiment integration pipeline.")
    parser.add_argument("--provider", choices=["newsapi", "gdelt"], default="newsapi")
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument(
        "--countries",
        nargs="*",
        help="Optional subset of countries. Defaults to all countries in the main dataset.",
    )
    parser.add_argument("--chunk-months", type=int, default=3, help="Months per API request window.")
    parser.add_argument("--page-size", type=int, default=100, help="Results per API page (NewsAPI only).")
    parser.add_argument("--max-pages", type=int, default=2, help="Maximum pages per request window (NewsAPI).")
    parser.add_argument("--gdelt-max-records", type=int, default=250, help="Max records per GDELT request.")
    parser.add_argument("--pause", type=float, default=1.0, help="Seconds to sleep between API requests.")
    parser.add_argument("--keywords", nargs="*", help="Override default keyword list.")
    parser.add_argument("--skip-fetch", action="store_true", help="Reuse existing raw headlines instead of calling APIs.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing raw headlines file instead of appending.")
    parser.add_argument("--disable-ssl-verify", action="store_true", help="Disable SSL verification (useful for firewalled envs).")
    parser.add_argument("--newsapi-key", help="Override NEWSAPI_KEY env variable.")
    parser.add_argument("--base-dataset", type=Path, default=BASE_DATASET_FILE)
    parser.add_argument("--raw-output", type=Path, default=RAW_NEWS_FILE)
    parser.add_argument("--sentiment-output", type=Path, default=SENTIMENT_SCORES_FILE)
    parser.add_argument("--final-output", type=Path, default=FINAL_TRAINING_FILE)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def get_countries(dataset_path: Path, overrides: Optional[Sequence[str]]) -> List[str]:
    """Derive the country list from the base dataset unless overrides are provided."""
    if overrides:
        return sorted(set(overrides))

    df = pd.read_csv(dataset_path)
    countries = sorted(c for c in df["Country"].dropna().unique())
    logging.info("Loaded %d countries from %s", len(countries), dataset_path)
    return countries


def chunk_year_into_ranges(year: int, months: int) -> List[Tuple[date, date]]:
    """Yield (start, end) date tuples that partition a year into consecutive windows."""
    ranges: List[Tuple[date, date]] = []
    for start_month in range(1, 13, months):
        end_month = min(start_month + months - 1, 12)
        end_day = calendar.monthrange(year, end_month)[1]
        ranges.append((date(year, start_month, 1), date(year, end_month, end_day)))
    return ranges


def build_query(keywords: Sequence[str], country: str) -> str:
    keyword_expr = " OR ".join(keywords)
    return f"({keyword_expr}) AND \"{country}\""


class BaseNewsClient:
    provider_name: str

    def fetch_articles(
        self,
        *,
        country: str,
        query: str,
        start_date: date,
        end_date: date,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError


class NewsAPIClient(BaseNewsClient):
    base_url = "https://newsapi.org/v2/everything"

    def __init__(
        self,
        api_key: str,
        page_size: int = 100,
        max_pages: int = 2,
        verify_ssl: bool = True,
    ) -> None:
        if not api_key:
            raise ValueError("NewsAPI client requires a valid API key.")
        self.api_key = api_key
        self.page_size = min(max(page_size, 1), 100)
        self.max_pages = max(max_pages, 1)
        self.session = requests.Session()
        self.verify_ssl = verify_ssl
        self.provider_name = "newsapi"

    def fetch_articles(
        self,
        *,
        country: str,
        query: str,
        start_date: date,
        end_date: date,
    ) -> List[Dict[str, Any]]:
        params = {
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": self.page_size,
            "q": query,
            "from": start_date.isoformat(),
            "to": end_date.isoformat(),
        }
        records: List[Dict[str, Any]] = []
        for page in range(1, self.max_pages + 1):
            params["page"] = page
            params["apiKey"] = self.api_key
            try:
                response = self.session.get(
                    self.base_url,
                    params=params,
                    timeout=60,
                    verify=self.verify_ssl,
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                logging.warning("NewsAPI request failed for %s %s-%s: %s", country, start_date, end_date, exc)
                break

            payload = response.json()
            if payload.get("status") != "ok":
                logging.warning("NewsAPI returned non-ok status: %s", payload)
                break

            articles = payload.get("articles", [])
            logging.debug("Fetched %d articles for %s (%s - %s, page %d)", len(articles), country, start_date, end_date, page)
            for article in articles:
                records.append(
                    {
                        "country": country,
                        "provider": self.provider_name,
                        "query": query,
                        "title": article.get("title"),
                        "description": article.get("description"),
                        "content": article.get("content"),
                        "source": (article.get("source") or {}).get("name"),
                        "author": article.get("author"),
                        "published_at": article.get("publishedAt"),
                        "url": article.get("url"),
                        "retrieved_at": datetime.utcnow().isoformat(),
                    }
                )

            total_results = payload.get("totalResults", 0)
            if page * self.page_size >= total_results:
                break

        return records


class GDELTClient(BaseNewsClient):
    base_url = "https://api.gdeltproject.org/api/v2/doc/doc"

    def __init__(self, max_records: int = 250, verify_ssl: bool = True) -> None:
        self.max_records = max(1, min(max_records, 250))
        self.session = requests.Session()
        self.verify_ssl = verify_ssl
        self.provider_name = "gdelt"

    @staticmethod
    def _format_timestamp(dt: date) -> str:
        return f"{dt:%Y%m%d}000000"

    @staticmethod
    def _parse_gdelt_datetime(dt_str: Optional[str]) -> Optional[str]:
        if not dt_str:
            return None
        try:
            parsed = datetime.strptime(dt_str, "%Y%m%d%H%M%S")
            return parsed.isoformat()
        except ValueError:
            return None

    def fetch_articles(
        self,
        *,
        country: str,
        query: str,
        start_date: date,
        end_date: date,
    ) -> List[Dict[str, Any]]:
        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": self.max_records,
            "sort": "DateDesc",
            "startdatetime": self._format_timestamp(start_date),
            "enddatetime": self._format_timestamp(end_date),
        }

        try:
            response = self.session.get(
                self.base_url,
                params=params,
                timeout=60,
                verify=self.verify_ssl,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            logging.warning("GDELT request failed for %s %s-%s: %s", country, start_date, end_date, exc)
            return []

        payload = response.json()
        articles = payload.get("articles", [])
        records: List[Dict[str, Any]] = []
        for article in articles:
            published_at = self._parse_gdelt_datetime(article.get("seendate"))
            records.append(
                {
                    "country": country,
                    "provider": self.provider_name,
                    "query": query,
                    "title": article.get("title"),
                    "description": article.get("excerpt"),
                    "content": article.get("document") or article.get("shareImage") or "",
                    "source": article.get("sourceCommonName"),
                    "author": article.get("sourceCountry"),
                    "published_at": published_at,
                    "url": article.get("url"),
                    "retrieved_at": datetime.utcnow().isoformat(),
                }
            )

        logging.debug("Fetched %d GDELT articles for %s (%s - %s)", len(records), country, start_date, end_date)
        return records


def build_client(args: argparse.Namespace) -> BaseNewsClient:
    if args.provider == "newsapi":
        api_key = args.newsapi_key or os.getenv("NEWSAPI_KEY")
        if not api_key:
            raise RuntimeError("NEWSAPI_KEY is required for the NewsAPI provider.")
        return NewsAPIClient(
            api_key=api_key,
            page_size=args.page_size,
            max_pages=args.max_pages,
            verify_ssl=not args.disable_ssl_verify,
        )

    if args.provider == "gdelt":
        return GDELTClient(
            max_records=args.gdelt_max_records,
            verify_ssl=not args.disable_ssl_verify,
        )

    raise ValueError(f"Unsupported provider: {args.provider}")


def fetch_news_articles(
    client: BaseNewsClient,
    countries: Sequence[str],
    keywords: Sequence[str],
    start_year: int,
    end_year: int,
    chunk_months: int,
    pause_seconds: float,
) -> pd.DataFrame:
    """Fetch news articles for each country/year window."""
    records: List[Dict[str, Any]] = []
    for country in countries:
        for year in range(start_year, end_year + 1):
            for start_date, end_date in chunk_year_into_ranges(year, chunk_months):
                query = build_query(keywords, country)
                articles = client.fetch_articles(
                    country=country,
                    query=query,
                    start_date=start_date,
                    end_date=end_date,
                )
                for article in articles:
                    article.setdefault("year_hint", year)
                    article.setdefault("window_start", start_date.isoformat())
                    article.setdefault("window_end", end_date.isoformat())
                records.extend(articles)
                if pause_seconds:
                    time.sleep(pause_seconds)
    if not records:
        logging.warning("No articles fetched. Ensure the API credentials and parameters are correct.")
    return pd.DataFrame(records)


def combine_raw_data(new_df: pd.DataFrame, raw_path: Path, overwrite: bool) -> pd.DataFrame:
    """Merge new fetch results with any existing raw data."""
    if raw_path.exists() and not overwrite:
        existing = pd.read_csv(raw_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df.copy()

    if combined.empty:
        logging.warning("Combined raw news dataframe is empty.")
        return combined

    dedup_cols = ["country", "title", "published_at", "url"]
    combined = combined.drop_duplicates(subset=dedup_cols, keep="last")
    combined.to_csv(raw_path, index=False)
    logging.info("Saved %d raw news articles to %s", len(combined), raw_path)
    return combined


def load_existing_raw(raw_path: Path) -> pd.DataFrame:
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw headlines file not found at {raw_path}.")
    logging.info("Loading existing raw headlines from %s", raw_path)
    return pd.read_csv(raw_path)


def _safe_parse_year(value: Any, fallback: Optional[int]) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return fallback
        return int(value)
    if isinstance(value, str):
        try:
            return pd.to_datetime(value, utc=True).year
        except (ValueError, TypeError):
            return fallback
    return fallback


def compute_sentiment(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute sentiment scores and aggregate yearly averages."""
    if raw_df.empty:
        raise ValueError("Raw dataframe is empty. Cannot compute sentiment.")

    analyzer = SentimentIntensityAnalyzer()
    df = raw_df.copy()
    for col in ["title", "description", "content"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    df["text"] = df[["title", "description", "content"]].agg(" ".join, axis=1).str.strip()
    df["vader_score"] = df["text"].apply(lambda text: analyzer.polarity_scores(text)["compound"] if text else None)
    df["textblob_score"] = df["text"].apply(lambda text: TextBlob(text).sentiment.polarity if text else None)
    df["sentiment_score"] = df[["vader_score", "textblob_score"]].mean(axis=1, skipna=True)

    if "published_at" not in df.columns:
        df["published_at"] = pd.NaT

    df["year"] = df.apply(
        lambda row: _safe_parse_year(row.get("published_at"), row.get("year_hint")), axis=1
    )
    df["year"] = df["year"].fillna(method="ffill").fillna(method="bfill")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    sentiment_df = (
        df.dropna(subset=["country", "year"])
        .groupby(["country", "year"], as_index=False)["sentiment_score"]
        .mean()
    )

    return df, sentiment_df


def save_sentiment_scores(sentiment_df: pd.DataFrame, output_path: Path) -> None:
    sentiment_df = sentiment_df.sort_values(["country", "year"])
    sentiment_df.to_csv(output_path, index=False)
    logging.info("Saved sentiment scores to %s", output_path)


def merge_with_main_dataset(
    main_dataset_path: Path,
    sentiment_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    main_df = pd.read_csv(main_dataset_path)
    merged = main_df.merge(
        sentiment_df,
        left_on=["Country", "Year"],
        right_on=["country", "year"],
        how="left",
    )
    merged = merged.drop(columns=["country", "year"])
    if "sentiment_score" not in merged.columns:
        merged["sentiment_score"] = pd.NA

    merged["sentiment_score"] = (
        merged.groupby("Country")["sentiment_score"]
        .transform(lambda col: col.ffill().bfill())
    )

    global_mean = merged["sentiment_score"].mean(skipna=True)
    if pd.isna(global_mean):
        global_mean = 0.0
    merged["sentiment_score"] = merged["sentiment_score"].fillna(global_mean)

    merged.to_csv(output_path, index=False)
    logging.info("Saved final training data with sentiment to %s", output_path)
    return merged


def main() -> int:
    load_dotenv()
    args = parse_args()
    configure_logging(args.log_level)
    ensure_directories()

    args.base_dataset = args.base_dataset.resolve()
    args.raw_output = args.raw_output.resolve()
    args.sentiment_output = args.sentiment_output.resolve()
    args.final_output = args.final_output.resolve()

    keywords = args.keywords or DEFAULT_KEYWORDS
    countries = get_countries(args.base_dataset, args.countries)

    if args.skip_fetch:
        raw_df = load_existing_raw(args.raw_output)
    else:
        client = build_client(args)
        fetched_df = fetch_news_articles(
            client=client,
            countries=countries,
            keywords=keywords,
            start_year=args.start_year,
            end_year=args.end_year,
            chunk_months=args.chunk_months,
            pause_seconds=args.pause,
        )
        raw_df = combine_raw_data(fetched_df, args.raw_output, args.overwrite)

    _, sentiment_scores = compute_sentiment(raw_df)
    save_sentiment_scores(sentiment_scores, args.sentiment_output)
    merge_with_main_dataset(args.base_dataset, sentiment_scores, args.final_output)
    return 0


if __name__ == "__main__":
    sys.exit(main())


