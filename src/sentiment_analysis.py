"""sentiment analysis for corruption risk prediction using gdelt news data."""

import argparse
import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import requests
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SENTIMENT_DIR = DATA_DIR / "sentiment"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINT_FILE = SENTIMENT_DIR / "fetch_progress.json"

# ensure directories exist
SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# default corruption-related keywords
DEFAULT_KEYWORDS = [
    "corruption",
    "bribery",
    '"money laundering"',
    "fraud",
    "embezzlement",
    "scandal"
]

# initialize sentiment analyzers
vader_analyzer = SentimentIntensityAnalyzer()

# country name mapping for gdelt api (some short names are rejected)
# maps original country name to api-compatible search term
COUNTRY_API_MAPPING = {
    "Iraq": "Iraqi"  # gdelt api requires phrases > 4 characters
}


def get_api_country_name(country: str) -> str:
    """get api-compatible country name for gdelt queries."""
    return COUNTRY_API_MAPPING.get(country, country)


class BaseNewsClient(ABC):
    """abstract base class for news article clients."""
    
    @abstractmethod
    def fetch_articles(
        self,
        country: str,
        query: str,
        start_date: datetime,
        end_date: datetime,
        max_records: int = 250
    ) -> List[Dict[str, Any]]:
        """fetch news articles for given parameters."""
        pass


class GDELTClient(BaseNewsClient):
    """client for gdelt doc api v2."""
    
    base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
    
    def fetch_articles(
        self,
        country: str,
        query: str,
        start_date: datetime,
        end_date: datetime,
        max_records: int = 250
    ) -> List[Dict[str, Any]]:
        """fetch articles from gdelt api.
        
        note: gdelt api v2 only supports dates from 2017 onwards.
        earlier dates will return an error and be skipped.
        """
        
        # gdelt v2 only supports dates from 2017
        if start_date.year < 2017:
            logger.warning(f"gdelt v2 only supports dates from 2017, skipping {country} {start_date.year}")
            return []
        
        # format dates for gdelt api (YYYYMMDDHHMMSS)
        start_str = start_date.strftime("%Y%m%d%H%M%S")
        end_str = end_date.strftime("%Y%m%d%H%M%S")
        
        # use api-compatible country name for query
        api_country = get_api_country_name(country)
        
        # build query with country and keywords
        full_query = f'({query}) AND "{api_country}"'
        
        params = {
            "query": full_query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": str(max_records),
            "startdatetime": start_str,
            "enddatetime": end_str,
            "sort": "date"
        }
        
        try:
            logger.info(f"fetching articles for {country} ({start_date.date()} to {end_date.date()})")
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            # check if response is valid json
            try:
                data = response.json()
            except ValueError:
                # response might be an error message
                error_text = response.text.strip()
                if "Invalid query" in error_text or "error" in error_text.lower():
                    logger.warning(f"api error for {country} {start_date.year}: {error_text[:100]}")
                    return []
                raise
            
            # gdelt returns articles in 'articles' key
            articles = data.get("articles", [])
            
            # extract relevant fields
            results = []
            for article in articles:
                results.append({
                    "country": country,
                    "headline": article.get("title", ""),
                    "url": article.get("url", ""),
                    "date": article.get("seendate", ""),
                    "snippet": article.get("snippet", "")
                })
            
            logger.info(f"fetched {len(results)} articles for {country}")
            return results
            
        except requests.exceptions.RequestException as e:
            logger.error(f"error fetching articles for {country}: {e}")
            return []
        except (KeyError, ValueError) as e:
            logger.error(f"error parsing response for {country}: {e}")
            return []


def build_query(keywords: Sequence[str]) -> str:
    """build or query from keywords."""
    return " OR ".join(keywords)


def chunk_year_into_ranges(year: int, chunk_months: int) -> List[Tuple[datetime, datetime]]:
    """split a year into date ranges of specified months."""
    ranges = []
    start = datetime(year, 1, 1)
    end_year = year
    
    while start.year == year:
        # calculate end date
        if chunk_months == 12:
            end = datetime(year, 12, 31, 23, 59, 59)
        else:
            # add months to start date
            if start.month + chunk_months > 12:
                end = datetime(year, 12, 31, 23, 59, 59)
            else:
                end = datetime(year, start.month + chunk_months, 1) - timedelta(seconds=1)
        
        ranges.append((start, end))
        
        # move to next chunk
        if end.year > year or end.month == 12:
            break
        start = datetime(year, end.month + 1, 1)
    
    return ranges


def calculate_sentiment(text: str) -> Tuple[float, float]:
    """calculate sentiment using vader and textblob."""
    if not text or not text.strip():
        return 0.0, 0.0
    
    # vader sentiment (compound score is already -1 to 1)
    vader_scores = vader_analyzer.polarity_scores(text)
    vader_score = vader_scores["compound"]
    
    # textblob sentiment (polarity is -1 to 1)
    blob = TextBlob(text)
    textblob_score = blob.sentiment.polarity
    
    return vader_score, textblob_score


def load_checkpoint() -> set:
    """load list of completed country-year combinations."""
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                data = json.load(f)
                return set(tuple(item) for item in data)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"error loading checkpoint: {e}, starting fresh")
            return set()
    return set()


def save_checkpoint(completed_set: set) -> None:
    """save progress to checkpoint file."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(list(completed_set), f)


def fetch_news_articles(
    client: BaseNewsClient,
    countries: Sequence[str],
    keywords: Sequence[str],
    start_year: int,
    end_year: int,
    chunk_months: int,
    pause_seconds: float,
    max_records: int,
    overwrite: bool = False
) -> pd.DataFrame:
    """fetch news articles for each country/year window."""
    
    records: List[Dict[str, Any]] = []
    completed = load_checkpoint() if not overwrite else set()
    
    query = build_query(keywords)
    
    for country in countries:
        for year in range(start_year, end_year + 1):
            # check if this country-year is already done
            if (country, year) in completed:
                logger.info(f"skipping {country} {year} (already fetched)")
                continue
            
            for start_date, end_date in chunk_year_into_ranges(year, chunk_months):
                articles = client.fetch_articles(
                    country=country,
                    query=query,
                    start_date=start_date,
                    end_date=end_date,
                    max_records=max_records
                )
                
                # add year and date info to articles
                for article in articles:
                    article["year"] = year
                    article["month"] = start_date.month
                    article["window_start"] = start_date.isoformat()
                    article["window_end"] = end_date.isoformat()
                
                records.extend(articles)
                
                if pause_seconds > 0:
                    time.sleep(pause_seconds)
            
            # mark this country-year as complete
            completed.add((country, year))
            save_checkpoint(completed)
            logger.info(f"✓ completed {country} {year}")
    
    return pd.DataFrame(records)


def process_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """calculate sentiment scores for articles."""
    logger.info("calculating sentiment scores...")
    
    # combine headline and snippet for sentiment analysis
    df["text"] = df["headline"].fillna("") + " " + df["snippet"].fillna("")
    
    # calculate sentiment
    sentiment_results = df["text"].apply(calculate_sentiment)
    df["sentiment_vader"] = sentiment_results.apply(lambda x: x[0])
    df["sentiment_textblob"] = sentiment_results.apply(lambda x: x[1])
    
    # average of both methods
    df["sentiment_score"] = (df["sentiment_vader"] + df["sentiment_textblob"]) / 2.0
    
    return df


def aggregate_sentiment_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """aggregate sentiment scores by country and year."""
    logger.info("aggregating sentiment by country and year...")
    
    if df.empty:
        logger.warning("no articles to aggregate")
        return pd.DataFrame(columns=["country", "year", "sentiment_score", "article_count"])
    
    aggregated = df.groupby(["country", "year"]).agg({
        "sentiment_score": "mean",
        "headline": "count"
    }).reset_index()
    
    aggregated.columns = ["country", "year", "sentiment_score", "article_count"]
    
    return aggregated


def merge_with_main_dataset(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """merge sentiment scores with main training dataset."""
    logger.info("merging sentiment with main dataset...")
    
    main_df_path = PROCESSED_DIR / "corruption_data_expanded_labeled.csv"
    
    if not main_df_path.exists():
        logger.error(f"main dataset not found at {main_df_path}")
        return pd.DataFrame()
    
    main_df = pd.read_csv(main_df_path)
    
    # ensure year is numeric for merging
    main_df["Year"] = pd.to_numeric(main_df["Year"], errors="coerce")
    sentiment_df["year"] = pd.to_numeric(sentiment_df["year"], errors="coerce")
    
    # merge on country and year
    merged_df = main_df.merge(
        sentiment_df[["country", "year", "sentiment_score", "article_count"]],
        left_on=["Country", "Year"],
        right_on=["country", "year"],
        how="left"
    )
    
    # drop duplicate columns
    merged_df = merged_df.drop(columns=["country", "year"], errors="ignore")
    
    # fill missing sentiment scores with 0 (neutral)
    merged_df["sentiment_score"] = merged_df["sentiment_score"].fillna(0.0)
    merged_df["article_count"] = merged_df["article_count"].fillna(0)
    
    logger.info(f"merged dataset: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
    
    return merged_df


def get_all_countries() -> List[str]:
    """get list of all countries from main dataset."""
    main_df_path = PROCESSED_DIR / "corruption_data_expanded_labeled.csv"
    
    if not main_df_path.exists():
        logger.warning("main dataset not found, using default country list")
        return [
            "Angola", "Australia", "Brazil", "Canada", "Denmark", "Germany",
            "India", "Iraq", "Malaysia", "Mozambique", "New Zealand", "Norway",
            "Philippines", "Singapore", "South Africa", "Switzerland", "Ukraine",
            "Venezuela", "Zimbabwe"
        ]
    
    df = pd.read_csv(main_df_path)
    return sorted(df["Country"].unique().tolist())


def main():
    """main entry point."""
    parser = argparse.ArgumentParser(
        description="collect sentiment data from gdelt for corruption risk prediction"
    )
    
    parser.add_argument(
        "--provider",
        type=str,
        default="gdelt",
        choices=["gdelt"],
        help="news provider (currently only gdelt supported)"
    )
    
    parser.add_argument(
        "--start-year",
        type=int,
        default=2010,
        help="start year for data collection"
    )
    
    parser.add_argument(
        "--end-year",
        type=int,
        default=2023,
        help="end year for data collection"
    )
    
    parser.add_argument(
        "--countries",
        type=str,
        nargs="+",
        default=None,
        help="list of countries to fetch (default: all countries in dataset)"
    )
    
    parser.add_argument(
        "--keywords",
        type=str,
        nargs="+",
        default=DEFAULT_KEYWORDS,
        help="search keywords for corruption-related articles"
    )
    
    parser.add_argument(
        "--chunk-months",
        type=int,
        default=3,
        help="number of months per time chunk"
    )
    
    parser.add_argument(
        "--pause",
        type=float,
        default=2.0,
        help="pause between api calls in seconds"
    )
    
    parser.add_argument(
        "--gdelt-max-records",
        type=int,
        default=250,
        help="maximum records per gdelt api call"
    )
    
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="skip fetching, only process existing raw data"
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing data and checkpoint"
    )
    
    args = parser.parse_args()
    
    # determine countries to process
    if args.countries:
        countries = args.countries
    else:
        countries = get_all_countries()
    
    logger.info(f"processing {len(countries)} countries: {countries}")
    logger.info(f"year range: {args.start_year} to {args.end_year}")
    logger.info(f"keywords: {args.keywords}")
    
    # initialize client
    if args.provider == "gdelt":
        client = GDELTClient()
    else:
        raise ValueError(f"unsupported provider: {args.provider}")
    
    # file paths
    raw_file = SENTIMENT_DIR / "news_headlines_raw.csv"
    sentiment_file = SENTIMENT_DIR / "sentiment_scores.csv"
    final_file = PROCESSED_DIR / "final_training_data.csv"
    
    # fetch articles (unless skipping)
    if not args.skip_fetch:
        if args.overwrite and raw_file.exists():
            logger.info("overwriting existing raw data...")
            raw_file.unlink()
            if CHECKPOINT_FILE.exists():
                CHECKPOINT_FILE.unlink()
        
        logger.info("fetching news articles...")
        articles_df = fetch_news_articles(
            client=client,
            countries=countries,
            keywords=args.keywords,
            start_year=args.start_year,
            end_year=args.end_year,
            chunk_months=args.chunk_months,
            pause_seconds=args.pause,
            max_records=args.gdelt_max_records,
            overwrite=args.overwrite
        )
        
        if articles_df.empty:
            logger.warning("no articles fetched")
            return
        
        # save raw articles
        if raw_file.exists() and not args.overwrite:
            # append and deduplicate
            existing_df = pd.read_csv(raw_file)
            combined_df = pd.concat([existing_df, articles_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=["country", "headline", "url"], keep="first")
            combined_df.to_csv(raw_file, index=False)
            logger.info(f"appended to {raw_file}")
        else:
            articles_df.to_csv(raw_file, index=False)
            logger.info(f"saved raw articles to {raw_file}")
        
        # process sentiment
        articles_df = process_sentiment(articles_df)
    else:
        # load existing raw data
        if not raw_file.exists():
            logger.error(f"raw data file not found: {raw_file}")
            return
        
        logger.info(f"loading existing raw data from {raw_file}")
        articles_df = pd.read_csv(raw_file)
        articles_df = process_sentiment(articles_df)
    
    # aggregate by country and year
    sentiment_df = aggregate_sentiment_by_year(articles_df)
    
    # save sentiment scores
    if sentiment_file.exists() and not args.overwrite and not args.skip_fetch:
        # merge with existing
        existing_sentiment = pd.read_csv(sentiment_file)
        combined_sentiment = pd.concat([existing_sentiment, sentiment_df], ignore_index=True)
        combined_sentiment = combined_sentiment.drop_duplicates(subset=["country", "year"], keep="last")
        combined_sentiment = combined_sentiment.sort_values(["country", "year"])
        combined_sentiment.to_csv(sentiment_file, index=False)
        logger.info(f"updated {sentiment_file}")
    else:
        sentiment_df.to_csv(sentiment_file, index=False)
        logger.info(f"saved sentiment scores to {sentiment_file}")
    
    # merge with main dataset
    final_df = merge_with_main_dataset(sentiment_df)
    
    if not final_df.empty:
        final_df.to_csv(final_file, index=False)
        logger.info(f"saved final training data to {final_file}")
        logger.info(f"final dataset: {final_df.shape[0]} rows, {final_df.shape[1]} columns")
        logger.info(f"countries with sentiment: {final_df[final_df['sentiment_score'] != 0]['Country'].nunique()}")
    else:
        logger.error("failed to create final dataset")


if __name__ == "__main__":
    main()

