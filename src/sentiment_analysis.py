"""sentiment analysis for corruption risk prediction using gdelt news data."""

import argparse
import json
import logging
import re
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
API_USAGE_FILE = SENTIMENT_DIR / "guardian_api_usage.json"

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
        
        # retry logic for ssl/timeout errors
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                logger.info(f"fetching articles for {country} ({start_date.date()} to {end_date.date()}) [attempt {attempt + 1}/{max_retries}]")
                
                # increase timeout for later attempts
                timeout = 30 + (attempt * 30)
                response = requests.get(self.base_url, params=params, timeout=timeout, verify=True)
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
                
            except requests.exceptions.SSLError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"ssl error for {country} (attempt {attempt + 1}), retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # exponential backoff
                else:
                    logger.error(f"ssl error for {country} after {max_retries} attempts: {e}")
                    return []
            except requests.exceptions.Timeout as e:
                if attempt < max_retries - 1:
                    logger.warning(f"timeout for {country} (attempt {attempt + 1}), retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # exponential backoff
                else:
                    logger.error(f"timeout for {country} after {max_retries} attempts: {e}")
                    return []
            except requests.exceptions.RequestException as e:
                logger.error(f"error fetching articles for {country}: {e}")
                return []
            except (KeyError, ValueError) as e:
                logger.error(f"error parsing response for {country}: {e}")
                return []
        
        # should not reach here, but just in case
        return []


def load_api_usage() -> Dict[str, Dict[str, Any]]:
    """load api key usage tracking."""
    if API_USAGE_FILE.exists():
        try:
            with open(API_USAGE_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"error loading api usage: {e}, starting fresh")
            return {}
    return {}


def save_api_usage(usage: Dict[str, Dict[str, Any]]) -> None:
    """save api key usage tracking."""
    with open(API_USAGE_FILE, 'w') as f:
        json.dump(usage, f, indent=2)


def get_available_api_key(api_keys: List[str], max_requests_per_key: int = 500) -> Optional[str]:
    """get next available api key, rotating when limit is reached."""
    usage = load_api_usage()
    today = datetime.now().strftime("%Y-%m-%d")
    
    for api_key in api_keys:
        key_id = api_key[:8]  # use first 8 chars as identifier
        if key_id not in usage:
            usage[key_id] = {"api_key": api_key, "requests": {}, "total_requests": 0}
        
        # check today's usage
        today_requests = usage[key_id]["requests"].get(today, 0)
        
        if today_requests < max_requests_per_key:
            return api_key
    
    # all keys exhausted
    logger.error(f"all api keys have reached daily limit ({max_requests_per_key} requests)")
    return None


def increment_api_usage(api_key: str) -> None:
    """increment usage counter for an api key."""
    usage = load_api_usage()
    today = datetime.now().strftime("%Y-%m-%d")
    key_id = api_key[:8]
    
    if key_id not in usage:
        usage[key_id] = {"api_key": api_key, "requests": {}, "total_requests": 0}
    
    usage[key_id]["requests"][today] = usage[key_id]["requests"].get(today, 0) + 1
    usage[key_id]["total_requests"] = usage[key_id].get("total_requests", 0) + 1
    
    save_api_usage(usage)
    
    # log usage
    today_requests = usage[key_id]["requests"][today]
    logger.debug(f"api key {key_id}...: {today_requests} requests today")


class GuardianClient(BaseNewsClient):
    """client for guardian api with multi-key support."""

    base_url = "https://content.guardianapis.com/search"
    default_api_keys = [
        "93c1ada7-8c9d-4b38-aa03-20b53f43a1cb",  # primary key
        "c1bb1584-e60d-4d7a-b019-276f96dd0a53",  # key 2
        "ea1edef5-f57f-4f1d-80fe-92b9a81b4db4",  # key 3
        "54917b96-22e5-4b8c-8c5b-faae17840526",  # key 4
        "7f5c4bbf-6d72-445a-85f4-399aef09861d",  # key 5
    ]
    
    # corruption keywords for relevance filtering
    corruption_keywords = [
        "corruption", "bribery", "fraud", "scandal", 
        "embezzlement", "money laundering", "laundering"
    ]
    
    def __init__(self, api_keys: Optional[List[str]] = None, max_requests_per_key: int = 500):
        """initialize guardian client with optional api keys.
        
        args:
            api_keys: list of api keys to use (rotates automatically)
            max_requests_per_key: max requests per key per day (default 500)
        """
        self.api_keys = api_keys or self.default_api_keys
        self.max_requests_per_key = max_requests_per_key
        self.current_api_key = None
    
    def _get_api_key(self) -> Optional[str]:
        """get available api key, rotating if needed."""
        api_key = get_available_api_key(self.api_keys, self.max_requests_per_key)
        if api_key and api_key != self.current_api_key:
            key_id = api_key[:8]
            logger.info(f"using guardian api key: {key_id}...")
            self.current_api_key = api_key
        return api_key
    
    def fetch_articles(
        self,
        country: str,
        query: str,
        start_date: datetime,
        end_date: datetime,
        max_records: int = 250
    ) -> List[Dict[str, Any]]:
        """fetch articles from guardian api with relevance filtering."""
        
        # guardian only supports dates up to 2016 reliably
        if start_date.year > 2016:
            logger.warning(f"guardian api works best for 2016 and earlier, skipping {country} {start_date.year}")
            return []
        
        # format dates for guardian api (YYYY-MM-DD)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        # build broader query (country + general corruption terms)
        # we'll filter for relevance client-side
        search_query = f"{country} AND (corruption OR bribery OR fraud OR scandal)"
        
        all_articles = []
        page = 1
        max_pages = 20  # guardian returns max 50 per page, so 20 pages = 1000 articles to filter
        
        while len(all_articles) < max_records and page <= max_pages:
            # get available api key (rotates automatically)
            api_key = self._get_api_key()
            if not api_key:
                logger.error(f"no available api keys, stopping collection for {country}")
                break
            
            params = {
                "api-key": api_key,
                "q": search_query,
                "from-date": start_str,
                "to-date": end_str,
                "page-size": 50,  # max allowed
                "page": page,
                "show-fields": "trailText,headline",
                "order-by": "relevance"
            }
            
            try:
                logger.info(f"fetching guardian page {page} for {country} ({start_date.date()} to {end_date.date()})")
                response = requests.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                
                # increment usage counter after successful request
                increment_api_usage(api_key)
                
                data = response.json()
                
                # guardian returns articles in response.results
                results = data.get("response", {}).get("results", [])
                
                if not results:
                    break  # no more articles
                
                # filter for relevance: since guardian api already filters by country + corruption terms,
                # we just verify that a corruption keyword appears in the text
                # (the api already ensured the country is mentioned, so we trust that)
                for article in results:
                    headline = article.get("webTitle", "").lower()
                    snippet_raw = article.get("fields", {}).get("trailText", "")
                    # remove html tags from snippet for better matching
                    snippet = re.sub(r'<[^>]+>', '', snippet_raw).lower() if snippet_raw else ""
                    combined_text = headline + " " + snippet
                    
                    # check if at least one corruption keyword appears
                    # (guardian api already filtered by country, so we trust that)
                    if not any(keyword in combined_text for keyword in self.corruption_keywords):
                        continue
                    
                    # article is relevant
                    all_articles.append({
                        "country": country,
                        "headline": article.get("webTitle", ""),
                        "url": article.get("webUrl", ""),
                        "date": article.get("webPublicationDate", ""),
                        "snippet": article.get("fields", {}).get("trailText", "")
                    })
                    
                    if len(all_articles) >= max_records:
                        break
                
                # check if we've reached the last page
                if page >= data.get("response", {}).get("pages", 1):
                    break
                
                page += 1
                
                # respect rate limit: 1 request per second
                time.sleep(1.1)
                
            except requests.exceptions.HTTPError as e:
                # check for rate limit errors (429) or auth errors (401, 403)
                if e.response.status_code == 429:
                    logger.warning(f"rate limit hit for api key {api_key[:8]}..., rotating to next key")
                    # mark this key as exhausted for today
                    usage = load_api_usage()
                    today = datetime.now().strftime("%Y-%m-%d")
                    key_id = api_key[:8]
                    if key_id in usage:
                        usage[key_id]["requests"][today] = self.max_requests_per_key
                        save_api_usage(usage)
                    # try next key
                    continue
                elif e.response.status_code in [401, 403]:
                    logger.error(f"authentication error for api key {api_key[:8]}...: {e}")
                    # skip this key and try next
                    continue
                else:
                    logger.error(f"http error fetching guardian articles for {country}: {e}")
                    break
            except requests.exceptions.RequestException as e:
                logger.error(f"error fetching guardian articles for {country}: {e}")
                break
            except (KeyError, ValueError) as e:
                logger.error(f"error parsing guardian response for {country}: {e}")
                break
        
        logger.info(f"fetched {len(all_articles)} relevant articles for {country} after filtering")
        return all_articles


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
    overwrite: bool = False,
    auto_provider: bool = True
) -> pd.DataFrame:
    """fetch news articles for each country/year window.
    
    if auto_provider is true, automatically selects:
    - guardian for years 2010-2016
    - gdelt for years 2017-2023
    """
    
    records: List[Dict[str, Any]] = []
    completed = load_checkpoint() if not overwrite else set()
    
    query = build_query(keywords)
    
    for country in countries:
        for year in range(start_year, end_year + 1):
            # check if this country-year is already done
            if (country, year) in completed:
                logger.info(f"skipping {country} {year} (already fetched)")
                continue
            
            # auto-select provider based on year
            if auto_provider:
                if year <= 2016:
                    # use guardian for historical data
                    # use default api keys (may have been updated in main())
                    year_client = GuardianClient()
                    logger.info(f"using guardian api for {country} {year}")
                else:
                    # use gdelt for modern data
                    year_client = GDELTClient()
                    logger.info(f"using gdelt api for {country} {year}")
            else:
                # use provided client
                year_client = client
            
            for start_date, end_date in chunk_year_into_ranges(year, chunk_months):
                articles = year_client.fetch_articles(
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
            
            # only mark as complete if we got some articles (or if it's a known empty year)
            # this prevents marking failed requests as complete
            articles_for_year = [a for a in records if a.get("year") == year]
            if len(articles_for_year) > 0 or year < 2017:  # guardian years might be legitimately empty
                completed.add((country, year))
                save_checkpoint(completed)
                logger.info(f"✓ completed {country} {year} ({len(articles_for_year)} articles)")
            else:
                logger.warning(f"⚠ no articles fetched for {country} {year}, not marking as complete (will retry)")
    
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
        default="auto",
        choices=["gdelt", "guardian", "auto"],
        help="news provider: 'gdelt', 'guardian', or 'auto' (selects based on year)"
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
    
    parser.add_argument(
        "--no-auto-provider",
        action="store_true",
        help="disable auto-provider selection (use --provider instead)"
    )
    
    parser.add_argument(
        "--guardian-api-keys",
        type=str,
        nargs="+",
        default=None,
        help="additional guardian api keys to use (rotates automatically when limit reached)"
    )
    
    parser.add_argument(
        "--guardian-max-requests",
        type=int,
        default=500,
        help="max requests per guardian api key per day (default 500)"
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
    
    # initialize client (only used if auto_provider is False)
    auto_provider = not args.no_auto_provider and args.provider == "auto"
    
    # prepare guardian api keys if provided
    guardian_api_keys = None
    if args.guardian_api_keys:
        guardian_api_keys = args.guardian_api_keys
        logger.info(f"using {len(guardian_api_keys)} guardian api key(s) with rotation")
    
    if args.provider == "gdelt":
        client = GDELTClient()
    elif args.provider == "guardian":
        client = GuardianClient(
            api_keys=guardian_api_keys,
            max_requests_per_key=args.guardian_max_requests
        )
    elif args.provider == "auto":
        # client will be selected per-year in fetch_news_articles
        # but we need to update GuardianClient instances to use the api keys
        # we'll do this by modifying the class default or passing through a global
        if guardian_api_keys:
            # update default api keys for all GuardianClient instances
            GuardianClient.default_api_keys = guardian_api_keys
        client = GDELTClient()  # placeholder, won't be used if auto_provider=True
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
            overwrite=args.overwrite,
            auto_provider=auto_provider
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

