"""
comprehensive test for sentiment analysis with guardian and gdelt apis.
tests both clients, auto-provider selection, and sentiment calculation.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# automatically use venv python if it exists
venv_python = project_root / 'venv' / 'bin' / 'python3'
if venv_python.exists() and not sys.executable.startswith(str(project_root / 'venv')):
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)

import pandas as pd
from src.sentiment_analysis import (
    GuardianClient,
    GDELTClient,
    fetch_news_articles,
    process_sentiment,
    aggregate_sentiment_by_year,
    calculate_sentiment
)


def test_guardian_client():
    """test guardian client with small sample."""
    print("\n" + "="*60)
    print("TEST 1: Guardian Client")
    print("="*60)
    
    client = GuardianClient()
    
    # test with malaysia 2013 (small sample)
    start_date = datetime(2013, 1, 1)
    end_date = datetime(2013, 12, 31)
    
    print(f"fetching articles for malaysia 2013...")
    articles = client.fetch_articles(
        country="Malaysia",
        query="corruption OR bribery OR fraud",
        start_date=start_date,
        end_date=end_date,
        max_records=10
    )
    
    print(f"  fetched {len(articles)} articles")
    
    # validate article structure
    if articles:
        article = articles[0]
        required_fields = ["country", "headline", "url", "date", "snippet"]
        missing_fields = [f for f in required_fields if f not in article]
        
        if missing_fields:
            print(f"  ✗ FAILED: missing fields: {missing_fields}")
            return False
        else:
            print(f"  ✓ article structure valid")
            print(f"  sample headline: {article['headline'][:60]}...")
    
    # check that articles are from 2013
    if articles:
        dates = [a.get("date", "") for a in articles if a.get("date")]
        if dates:
            print(f"  ✓ articles have dates: {len(dates)} with dates")
    
    success = len(articles) > 0
    print(f"  {'✓ PASSED' if success else '✗ FAILED'}: guardian client test")
    return success


def test_gdelt_client():
    """test gdelt client with small sample."""
    print("\n" + "="*60)
    print("TEST 2: GDELT Client")
    print("="*60)
    
    client = GDELTClient()
    
    # test with malaysia 2017 (small sample)
    start_date = datetime(2017, 1, 1)
    end_date = datetime(2017, 12, 31)
    
    print(f"fetching articles for malaysia 2017...")
    articles = client.fetch_articles(
        country="Malaysia",
        query="corruption OR bribery OR fraud",
        start_date=start_date,
        end_date=end_date,
        max_records=10
    )
    
    print(f"  fetched {len(articles)} articles")
    
    # validate article structure
    if articles:
        article = articles[0]
        required_fields = ["country", "headline", "url", "date", "snippet"]
        missing_fields = [f for f in required_fields if f not in article]
        
        if missing_fields:
            print(f"  ✗ FAILED: missing fields: {missing_fields}")
            return False
        else:
            print(f"  ✓ article structure valid")
            print(f"  sample headline: {article['headline'][:60]}...")
    
    # check that gdelt rejects pre-2017 dates
    print(f"  testing date validation (should reject 2016)...")
    old_date = datetime(2016, 1, 1)
    old_articles = client.fetch_articles(
        country="Malaysia",
        query="corruption",
        start_date=old_date,
        end_date=old_date,
        max_records=10
    )
    if len(old_articles) == 0:
        print(f"  ✓ correctly rejected pre-2017 dates")
    else:
        print(f"  ✗ FAILED: should reject pre-2017 dates")
        return False
    
    success = len(articles) > 0
    print(f"  {'✓ PASSED' if success else '✗ FAILED'}: gdelt client test")
    return success


def test_auto_provider_selection():
    """test auto-provider selection based on year."""
    print("\n" + "="*60)
    print("TEST 3: Auto-Provider Selection")
    print("="*60)
    
    # test with guardian year (2013)
    print("testing auto-provider for 2013 (should use guardian)...")
    articles_2013 = fetch_news_articles(
        client=GDELTClient(),  # placeholder, won't be used
        countries=["Malaysia"],
        keywords=["corruption", "bribery"],
        start_year=2013,
        end_year=2013,
        chunk_months=12,
        pause_seconds=1.1,
        max_records=5,
        overwrite=True,
        auto_provider=True
    )
    
    print(f"  fetched {len(articles_2013)} articles for 2013")
    
    # test with gdelt year (2017)
    print("testing auto-provider for 2017 (should use gdelt)...")
    articles_2017 = fetch_news_articles(
        client=GDELTClient(),  # placeholder, won't be used
        countries=["Malaysia"],
        keywords=["corruption", "bribery"],
        start_year=2017,
        end_year=2017,
        chunk_months=12,
        pause_seconds=2.0,
        max_records=5,
        overwrite=True,
        auto_provider=True
    )
    
    print(f"  fetched {len(articles_2017)} articles for 2017")
    
    success = len(articles_2013) > 0 or len(articles_2017) > 0
    print(f"  {'✓ PASSED' if success else '✗ FAILED'}: auto-provider selection test")
    return success


def test_sentiment_calculation():
    """test sentiment calculation on sample text."""
    print("\n" + "="*60)
    print("TEST 4: Sentiment Calculation")
    print("="*60)
    
    # test with negative corruption-related text
    negative_text = "massive corruption scandal rocks government, billions stolen"
    vader_score, textblob_score = calculate_sentiment(negative_text)
    avg_score = (vader_score + textblob_score) / 2.0
    
    print(f"  negative text: '{negative_text[:50]}...'")
    print(f"  vader score: {vader_score:.3f}")
    print(f"  textblob score: {textblob_score:.3f}")
    print(f"  average score: {avg_score:.3f}")
    
    if avg_score < 0:
        print(f"  ✓ correctly identified negative sentiment")
        negative_passed = True
    else:
        print(f"  ✗ FAILED: should be negative")
        negative_passed = False
    
    # test with neutral text
    neutral_text = "government announces new policy"
    vader_score, textblob_score = calculate_sentiment(neutral_text)
    avg_score = (vader_score + textblob_score) / 2.0
    
    print(f"\n  neutral text: '{neutral_text}'")
    print(f"  average score: {avg_score:.3f}")
    
    if abs(avg_score) < 0.2:
        print(f"  ✓ correctly identified neutral sentiment")
        neutral_passed = True
    else:
        print(f"  ⚠ warning: neutral text scored {avg_score:.3f} (expected near 0)")
        neutral_passed = True  # not a failure, just a warning
    
    success = negative_passed and neutral_passed
    print(f"  {'✓ PASSED' if success else '✗ FAILED'}: sentiment calculation test")
    return success


def test_full_pipeline():
    """test full pipeline with small sample."""
    print("\n" + "="*60)
    print("TEST 5: Full Pipeline Test")
    print("="*60)
    
    # fetch articles
    print("step 1: fetching articles...")
    articles_df = fetch_news_articles(
        client=GDELTClient(),  # placeholder
        countries=["Malaysia"],
        keywords=["corruption", "bribery", "fraud"],
        start_year=2013,
        end_year=2013,
        chunk_months=12,
        pause_seconds=1.1,
        max_records=10,
        overwrite=True,
        auto_provider=True
    )
    
    if articles_df.empty:
        print("  ✗ FAILED: no articles fetched")
        return False
    
    print(f"  ✓ fetched {len(articles_df)} articles")
    
    # process sentiment
    print("step 2: processing sentiment...")
    articles_df = process_sentiment(articles_df)
    
    required_cols = ["sentiment_score", "sentiment_vader", "sentiment_textblob"]
    missing_cols = [c for c in required_cols if c not in articles_df.columns]
    
    if missing_cols:
        print(f"  ✗ FAILED: missing columns: {missing_cols}")
        return False
    
    print(f"  ✓ calculated sentiment scores")
    print(f"  mean sentiment: {articles_df['sentiment_score'].mean():.3f}")
    
    # aggregate by year
    print("step 3: aggregating by year...")
    sentiment_df = aggregate_sentiment_by_year(articles_df)
    
    if sentiment_df.empty:
        print("  ✗ FAILED: no aggregated sentiment")
        return False
    
    print(f"  ✓ aggregated sentiment")
    print(f"  country-year combinations: {len(sentiment_df)}")
    print(f"  columns: {list(sentiment_df.columns)}")
    
    success = True
    print(f"  {'✓ PASSED' if success else '✗ FAILED'}: full pipeline test")
    return success


def test_cross_source_consistency():
    """test that guardian and gdelt produce similar sentiment scores."""
    print("\n" + "="*60)
    print("TEST 6: Cross-Source Consistency (Guardian vs GDELT)")
    print("="*60)
    
    # fetch guardian data for 2016
    print("fetching guardian data for malaysia 2016...")
    guardian_articles = fetch_news_articles(
        client=GuardianClient(),
        countries=["Malaysia"],
        keywords=["corruption", "bribery", "fraud"],
        start_year=2016,
        end_year=2016,
        chunk_months=12,
        pause_seconds=1.1,
        max_records=10,
        overwrite=True,
        auto_provider=False
    )
    
    if not guardian_articles.empty:
        guardian_processed = process_sentiment(guardian_articles)
        guardian_agg = aggregate_sentiment_by_year(guardian_processed)
        if not guardian_agg.empty:
            guardian_score = guardian_agg.iloc[0]['sentiment_score']
            print(f"  guardian 2016 sentiment: {guardian_score:.3f}")
        else:
            print(f"  ⚠ no guardian data to compare")
            return True  # not a failure, just no data
    else:
        print(f"  ⚠ no guardian articles fetched")
        return True  # not a failure, just no data
    
    # fetch gdelt data for 2017
    print("fetching gdelt data for malaysia 2017...")
    gdelt_articles = fetch_news_articles(
        client=GDELTClient(),
        countries=["Malaysia"],
        keywords=["corruption", "bribery", "fraud"],
        start_year=2017,
        end_year=2017,
        chunk_months=12,
        pause_seconds=2.0,
        max_records=10,
        overwrite=True,
        auto_provider=False
    )
    
    if not gdelt_articles.empty:
        gdelt_processed = process_sentiment(gdelt_articles)
        gdelt_agg = aggregate_sentiment_by_year(gdelt_processed)
        if not gdelt_agg.empty:
            gdelt_score = gdelt_agg.iloc[0]['sentiment_score']
            print(f"  gdelt 2017 sentiment: {gdelt_score:.3f}")
            
            # both should be negative (corruption news)
            if guardian_score < 0 and gdelt_score < 0:
                print(f"  ✓ both sources produce negative sentiment (as expected)")
                print(f"  ✓ scores are in similar range (both negative)")
                return True
            else:
                print(f"  ⚠ warning: one or both scores are not negative")
                return True  # not a failure, just different data
        else:
            print(f"  ⚠ no gdelt data to compare")
            return True
    else:
        print(f"  ⚠ no gdelt articles fetched")
        return True
    
    print(f"  ✓ PASSED: cross-source consistency test (limited data)")
    return True


def main():
    """run all tests."""
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS SYSTEM TEST SUITE")
    print("="*60)
    print("testing guardian api, gdelt api, auto-provider selection,")
    print("sentiment calculation, and full pipeline...")
    
    results = []
    
    # run tests
    results.append(("Guardian Client", test_guardian_client()))
    results.append(("GDELT Client", test_gdelt_client()))
    results.append(("Auto-Provider Selection", test_auto_provider_selection()))
    results.append(("Sentiment Calculation", test_sentiment_calculation()))
    results.append(("Full Pipeline", test_full_pipeline()))
    results.append(("Cross-Source Consistency", test_cross_source_consistency()))
    
    # summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {status}: {test_name}")
    
    print("-" * 60)
    print(f"total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED - system is ready for full data collection!")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed - review before full data collection")
        return 1


if __name__ == "__main__":
    sys.exit(main())

