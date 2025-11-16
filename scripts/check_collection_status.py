#!/usr/bin/env python3
"""
quick status check for sentiment data collection.
shows progress, checkpoint status, and data counts.
"""

import json
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
checkpoint_file = project_root / "data" / "sentiment" / "fetch_progress.json"
sentiment_file = project_root / "data" / "sentiment" / "sentiment_scores.csv"
raw_file = project_root / "data" / "sentiment" / "news_headlines_raw.csv"

print("="*60)
print("SENTIMENT DATA COLLECTION STATUS")
print("="*60)

# checkpoint status
if checkpoint_file.exists():
    with open(checkpoint_file, 'r') as f:
        completed = json.load(f)
    print(f"\n✓ Checkpoint file exists")
    print(f"  Completed country-years: {len(completed)}")
    
    # group by country
    countries = {}
    for country, year in completed:
        if country not in countries:
            countries[country] = []
        countries[country].append(year)
    
    print(f"  Countries with data: {len(countries)}")
    print(f"\n  Progress by country:")
    for country in sorted(countries.keys()):
        years = sorted(countries[country])
        year_range = f"{min(years)}-{max(years)}" if len(years) > 1 else str(years[0])
        print(f"    {country:20} {len(years):2} years ({year_range})")
else:
    print(f"\n⚠ Checkpoint file not found")

# sentiment scores
if sentiment_file.exists():
    df = pd.read_csv(sentiment_file)
    print(f"\n✓ Sentiment scores file exists")
    print(f"  Total country-years: {len(df)}")
    print(f"  Countries: {df['country'].nunique()}")
    print(f"  Year range: {df['year'].min()}-{df['year'].max()}")
    print(f"  Total articles: {df['article_count'].sum():,}")
    print(f"  Mean sentiment: {df['sentiment_score'].mean():.3f}")
    
    # check guardian vs gdelt
    guardian_years = df[df['year'] <= 2016]
    gdelt_years = df[df['year'] >= 2017]
    
    print(f"\n  Data source breakdown:")
    print(f"    Guardian (2010-2016): {len(guardian_years)} country-years, {guardian_years['article_count'].sum():,} articles")
    print(f"    GDELT (2017-2023):    {len(gdelt_years)} country-years, {gdelt_years['article_count'].sum():,} articles")
else:
    print(f"\n⚠ Sentiment scores file not found")

# raw articles
if raw_file.exists():
    df = pd.read_csv(raw_file)
    print(f"\n✓ Raw articles file exists")
    print(f"  Total articles: {len(df):,}")
    print(f"  Countries: {df['country'].nunique()}")
    print(f"  Year range: {df['year'].min()}-{df['year'].max()}")
else:
    print(f"\n⚠ Raw articles file not found")

print("\n" + "="*60)
print("Collection processes should be running in background.")
print("Check logs: data/sentiment/collection_log_guardian.txt")
print("            data/sentiment/collection_log_gdelt.txt")
print("="*60)

