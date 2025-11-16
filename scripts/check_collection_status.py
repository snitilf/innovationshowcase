#!/usr/bin/env python3
"""
quick status check for sentiment data collection.
shows progress, checkpoint status, and data counts with expected targets.
"""

import json
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
checkpoint_file = project_root / "data" / "sentiment" / "fetch_progress.json"
sentiment_file = project_root / "data" / "sentiment" / "sentiment_scores.csv"
raw_file = project_root / "data" / "sentiment" / "news_headlines_raw.csv"

# expected targets
TOTAL_COUNTRIES = 19
GUARDIAN_YEARS = list(range(2010, 2017))  # 2010-2016 = 7 years
GDELT_YEARS = list(range(2017, 2024))  # 2017-2023 = 7 years
EXPECTED_GUARDIAN = TOTAL_COUNTRIES * len(GUARDIAN_YEARS)  # 19 × 7 = 133
EXPECTED_GDELT = TOTAL_COUNTRIES * len(GDELT_YEARS)  # 19 × 7 = 133
EXPECTED_TOTAL = EXPECTED_GUARDIAN + EXPECTED_GDELT  # 266

print("="*70)
print("SENTIMENT DATA COLLECTION STATUS")
print("="*70)
print(f"\nExpected Coverage: 2010-2023 (14 years × {TOTAL_COUNTRIES} countries = {EXPECTED_TOTAL} country-years)")
print(f"  Guardian (2010-2016): {EXPECTED_GUARDIAN} country-years")
print(f"  GDELT (2017-2023):    {EXPECTED_GDELT} country-years")

# checkpoint status
if checkpoint_file.exists():
    with open(checkpoint_file, 'r') as f:
        completed = json.load(f)
    
    print(f"\n" + "="*70)
    print("CHECKPOINT STATUS")
    print("="*70)
    print(f"Total completed: {len(completed)} country-years")
    
    # separate guardian and gdelt
    guardian_completed = [(c, y) for c, y in completed if y <= 2016]
    gdelt_completed = [(c, y) for c, y in completed if y >= 2017]
    
    print(f"\nGuardian (2010-2016):")
    print(f"  Completed: {len(guardian_completed)}/{EXPECTED_GUARDIAN} country-years ({len(guardian_completed)/EXPECTED_GUARDIAN*100:.1f}%)")
    
    print(f"\nGDELT (2017-2023):")
    print(f"  Completed: {len(gdelt_completed)}/{EXPECTED_GDELT} country-years ({len(gdelt_completed)/EXPECTED_GDELT*100:.1f}%)")
    
    # group by country for guardian
    guardian_by_country = {}
    for country, year in guardian_completed:
        if country not in guardian_by_country:
            guardian_by_country[country] = []
        guardian_by_country[country].append(year)
    
    # group by country for gdelt
    gdelt_by_country = {}
    for country, year in gdelt_completed:
        if country not in gdelt_by_country:
            gdelt_by_country[country] = []
        gdelt_by_country[country].append(year)
    
    print(f"\nGuardian Progress by Country (2010-2016):")
    for country in sorted(guardian_by_country.keys()):
        years = sorted(guardian_by_country[country])
        expected = len(GUARDIAN_YEARS)
        missing = [y for y in GUARDIAN_YEARS if y not in years]
        status = "✓" if len(years) == expected else f"⚠ {len(missing)} missing"
        print(f"  {country:20} {len(years)}/{expected} years {status}")
        if missing and len(missing) <= 5:
            print(f"    Missing: {missing}")
    
    print(f"\nGDELT Progress by Country (2017-2023):")
    if gdelt_by_country:
        for country in sorted(gdelt_by_country.keys()):
            years = sorted(gdelt_by_country[country])
            expected = len(GDELT_YEARS)
            missing = [y for y in GDELT_YEARS if y not in years]
            status = "✓" if len(years) == expected else f"⚠ {len(missing)} missing"
            print(f"  {country:20} {len(years)}/{expected} years {status}")
            if missing and len(missing) <= 5:
                print(f"    Missing: {missing}")
    else:
        print("  No GDELT data collected yet")
    
    # show countries with no data
    all_countries = set()
    for country, year in completed:
        all_countries.add(country)
    
    # get expected countries list
    try:
        import sys
        sys.path.insert(0, str(project_root))
        from src.sentiment_analysis import get_all_countries
        expected_countries = set(get_all_countries())
        missing_countries_guardian = expected_countries - set(guardian_by_country.keys())
        missing_countries_gdelt = expected_countries - set(gdelt_by_country.keys())
        
        if missing_countries_guardian:
            print(f"\nCountries with no Guardian data: {sorted(missing_countries_guardian)}")
        if missing_countries_gdelt:
            print(f"Countries with no GDELT data: {sorted(missing_countries_gdelt)}")
    except:
        pass
else:
    print(f"\n⚠ Checkpoint file not found")

# raw articles data
if raw_file.exists():
    df = pd.read_csv(raw_file)
    print(f"\n" + "="*70)
    print("RAW ARTICLES DATA")
    print("="*70)
    print(f"Total articles: {len(df):,}")
    print(f"Countries: {df['country'].nunique()}")
    
    if len(df) > 0:
        print(f"Year range: {int(df['year'].min())}-{int(df['year'].max())}")
        
        # breakdown by source
        guardian_articles = df[df['year'] <= 2016]
        gdelt_articles = df[df['year'] >= 2017]
        
        print(f"\nBy data source:")
        print(f"  Guardian (2010-2016): {len(guardian_articles):,} articles from {guardian_articles['country'].nunique()} countries")
        print(f"  GDELT (2017-2023):    {len(gdelt_articles):,} articles from {gdelt_articles['country'].nunique()} countries")
        
        # articles per country-year
        guardian_by_cy = guardian_articles.groupby(['country', 'year']).size()
        gdelt_by_cy = gdelt_articles.groupby(['country', 'year']).size()
        
        print(f"\nArticles per country-year:")
        print(f"  Guardian: min={guardian_by_cy.min()}, max={guardian_by_cy.max()}, mean={guardian_by_cy.mean():.1f}")
        print(f"  GDELT:    min={gdelt_by_cy.min() if len(gdelt_by_cy) > 0 else 0}, max={gdelt_by_cy.max() if len(gdelt_by_cy) > 0 else 0}, mean={gdelt_by_cy.mean():.1f if len(gdelt_by_cy) > 0 else 0:.1f}")
else:
    print(f"\n⚠ Raw articles file not found")

# sentiment scores data
if sentiment_file.exists():
    df = pd.read_csv(sentiment_file)
    print(f"\n" + "="*70)
    print("SENTIMENT SCORES DATA")
    print("="*70)
    print(f"Total country-years: {len(df)}")
    print(f"Countries: {df['country'].nunique()}")
    
    if len(df) > 0:
        print(f"Year range: {int(df['year'].min())}-{int(df['year'].max())}")
        print(f"Total articles: {df['article_count'].sum():,}")
        print(f"Mean sentiment: {df['sentiment_score'].mean():.3f}")
        
        # breakdown by source
        guardian_scores = df[df['year'] <= 2016]
        gdelt_scores = df[df['year'] >= 2017]
        
        print(f"\nBy data source:")
        print(f"  Guardian (2010-2016): {len(guardian_scores)} country-years, {guardian_scores['article_count'].sum():,} articles")
        print(f"    Mean sentiment: {guardian_scores['sentiment_score'].mean():.3f}")
        print(f"  GDELT (2017-2023):    {len(gdelt_scores)} country-years, {gdelt_scores['article_count'].sum():,} articles")
        print(f"    Mean sentiment: {gdelt_scores['sentiment_score'].mean():.3f}")
        
        # coverage comparison
        print(f"\nCoverage vs Expected:")
        print(f"  Guardian: {len(guardian_scores)}/{EXPECTED_GUARDIAN} country-years ({len(guardian_scores)/EXPECTED_GUARDIAN*100:.1f}%)")
        print(f"  GDELT:    {len(gdelt_scores)}/{EXPECTED_GDELT} country-years ({len(gdelt_scores)/EXPECTED_GDELT*100:.1f}%)")
        print(f"  Total:    {len(df)}/{EXPECTED_TOTAL} country-years ({len(df)/EXPECTED_TOTAL*100:.1f}%)")
else:
    print(f"\n⚠ Sentiment scores file not found")

print("\n" + "="*70)
print("Collection processes should be running in background.")
print("Check logs: data/sentiment/collection_log_guardian_restart.txt")
print("            data/sentiment/collection_log_gdelt_restart.txt")
print("="*70)
