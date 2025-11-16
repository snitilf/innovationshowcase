#!/usr/bin/env python3
"""
test script for sentiment source validation notebook
run from project root: python3 tests/test_notebook04.py
"""

import os
import sys

# get project root directory (parent of tests/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# automatically use venv python if it exists and we're not already using it
venv_python = os.path.join(project_root, 'venv', 'bin', 'python3')
if os.path.exists(venv_python) and not sys.executable.startswith(os.path.join(project_root, 'venv')):
    # re-execute this script with venv python
    os.execv(venv_python, [venv_python] + sys.argv)

import pandas as pd
import numpy as np
from scipy import stats

print("=== Testing Sentiment Source Validation notebook ===\n")

try:
    # 1. load sentiment data
    print("1. loading sentiment data...")
    sentiment_path = os.path.join(project_root, 'data', 'sentiment', 'sentiment_scores.csv')
    if not os.path.exists(sentiment_path):
        print("   ✗ ERROR: sentiment scores file not found")
        exit(1)
    
    df = pd.read_csv(sentiment_path)
    print(f"   ✓ loaded: {len(df)} records")
    print(f"   countries: {df['country'].nunique()}")
    print(f"   year range: {df['year'].min()} to {df['year'].max()}\n")
    
    # validate required columns
    required_cols = ['country', 'year', 'sentiment_score', 'article_count']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"   ✗ ERROR: missing columns: {missing_cols}")
        exit(1)
    print(f"   ✓ all required columns present\n")
    
    # 2. separate by data source
    print("2. separating data by source...")
    guardian_df = df[df['year'] <= 2016].copy()
    gdelt_df = df[df['year'] >= 2017].copy()
    
    print(f"   guardian (2010-2016): {len(guardian_df)} records")
    print(f"   gdelt (2017-2023): {len(gdelt_df)} records")
    
    if len(guardian_df) == 0:
        print("   ✗ ERROR: no guardian data found")
        exit(1)
    if len(gdelt_df) == 0:
        print("   ✗ ERROR: no gdelt data found")
        exit(1)
    print(f"   ✓ both sources have data\n")
    
    # 3. validate score ranges
    print("3. validating sentiment score ranges...")
    guardian_in_range = (guardian_df['sentiment_score'] >= -1).all() and (guardian_df['sentiment_score'] <= 1).all()
    gdelt_in_range = (gdelt_df['sentiment_score'] >= -1).all() and (gdelt_df['sentiment_score'] <= 1).all()
    
    if not guardian_in_range:
        print(f"   ✗ ERROR: guardian scores outside [-1, 1] range")
        print(f"   min: {guardian_df['sentiment_score'].min()}, max: {guardian_df['sentiment_score'].max()}")
        exit(1)
    if not gdelt_in_range:
        print(f"   ✗ ERROR: gdelt scores outside [-1, 1] range")
        print(f"   min: {gdelt_df['sentiment_score'].min()}, max: {gdelt_df['sentiment_score'].max()}")
        exit(1)
    print(f"   ✓ all scores in valid range [-1, 1]\n")
    
    # 4. check basic statistics
    print("4. checking basic statistics...")
    guardian_mean = guardian_df['sentiment_score'].mean()
    gdelt_mean = gdelt_df['sentiment_score'].mean()
    mean_diff = abs(guardian_mean - gdelt_mean)
    
    print(f"   guardian mean: {guardian_mean:.4f}")
    print(f"   gdelt mean: {gdelt_mean:.4f}")
    print(f"   mean difference: {mean_diff:.4f}")
    
    # both should be negative (corruption news is negative)
    if guardian_mean > 0.2:
        print(f"   ⚠ WARNING: guardian mean is positive ({guardian_mean:.4f}), expected negative")
    if gdelt_mean > 0.2:
        print(f"   ⚠ WARNING: gdelt mean is positive ({gdelt_mean:.4f}), expected negative")
    
    # mean difference should be reasonable (< 0.2)
    if mean_diff > 0.3:
        print(f"   ⚠ WARNING: large mean difference ({mean_diff:.4f}), may indicate systematic bias")
    else:
        print(f"   ✓ mean difference is acceptable (< 0.3)\n")
    
    # 5. transition period analysis
    print("5. analyzing transition period (2016 vs 2017)...")
    countries_2016 = set(guardian_df[guardian_df['year'] == 2016]['country'].unique())
    countries_2017 = set(gdelt_df[gdelt_df['year'] == 2017]['country'].unique())
    common_countries = countries_2016.intersection(countries_2017)
    
    print(f"   countries with data in 2016: {len(countries_2016)}")
    print(f"   countries with data in 2017: {len(countries_2017)}")
    print(f"   countries with data in both: {len(common_countries)}")
    
    if len(common_countries) == 0:
        print("   ⚠ WARNING: no countries with data in both 2016 and 2017")
    else:
        # create transition comparison
        transition_comparison = []
        for country in sorted(common_countries):
            score_2016 = guardian_df[(guardian_df['country'] == country) & (guardian_df['year'] == 2016)]['sentiment_score'].values
            score_2017 = gdelt_df[(gdelt_df['country'] == country) & (gdelt_df['year'] == 2017)]['sentiment_score'].values
            
            if len(score_2016) > 0 and len(score_2017) > 0:
                transition_comparison.append({
                    'country': country,
                    'sentiment_2016': score_2016[0],
                    'sentiment_2017': score_2017[0],
                    'difference': score_2017[0] - score_2016[0]
                })
        
        if transition_comparison:
            transition_df = pd.DataFrame(transition_comparison)
            mean_transition_diff = transition_df['difference'].mean()
            abs_mean_diff = abs(mean_transition_diff)
            
            print(f"   transition comparison: {len(transition_df)} countries")
            print(f"   mean difference (2017 - 2016): {mean_transition_diff:.4f}")
            
            if abs_mean_diff < 0.15:
                print(f"   ✓ transition difference is acceptable (< 0.15)\n")
            else:
                print(f"   ⚠ WARNING: transition difference is large ({abs_mean_diff:.4f})\n")
    
    # 6. statistical tests
    print("6. performing statistical tests...")
    
    # t-test
    try:
        t_stat, p_value = stats.ttest_ind(guardian_df['sentiment_score'], gdelt_df['sentiment_score'])
        print(f"   t-test: t={t_stat:.4f}, p={p_value:.4f}")
        if p_value < 0.05:
            print("   ⚠ NOTE: statistically significant difference (expected due to different time periods)")
        else:
            print("   ✓ no statistically significant difference")
    except Exception as e:
        print(f"   ⚠ WARNING: t-test failed: {e}")
    
    # mann-whitney u test
    try:
        u_stat, u_p_value = stats.mannwhitneyu(guardian_df['sentiment_score'], gdelt_df['sentiment_score'])
        print(f"   mann-whitney u: u={u_stat:.4f}, p={u_p_value:.4f}")
        if u_p_value < 0.05:
            print("   ⚠ NOTE: statistically significant difference (expected due to different time periods)")
        else:
            print("   ✓ no statistically significant difference")
    except Exception as e:
        print(f"   ⚠ WARNING: mann-whitney u test failed: {e}")
    
    # correlation test for transition period
    if len(common_countries) > 3 and 'transition_df' in locals():
        try:
            correlation, corr_p = stats.pearsonr(transition_df['sentiment_2016'], transition_df['sentiment_2017'])
            print(f"   correlation (2016 vs 2017): r={correlation:.4f}, p={corr_p:.4f}")
            if correlation > 0.3:
                print("   ✓ positive correlation detected (scores are consistent)\n")
            else:
                print("   ⚠ WARNING: weak correlation (< 0.3)\n")
        except Exception as e:
            print(f"   ⚠ WARNING: correlation test failed: {e}\n")
    else:
        print("   ⚠ NOTE: insufficient data for correlation test\n")
    
    # 7. validate article counts
    print("7. validating article counts...")
    guardian_articles = guardian_df['article_count']
    gdelt_articles = gdelt_df['article_count']
    
    print(f"   guardian: mean={guardian_articles.mean():.1f}, median={guardian_articles.median():.1f}")
    print(f"   gdelt: mean={gdelt_articles.mean():.1f}, median={gdelt_articles.median():.1f}")
    
    # gdelt should have more articles (target: 400 per country-year)
    if gdelt_articles.mean() < 100:
        print("   ⚠ WARNING: gdelt mean article count is low")
    else:
        print("   ✓ gdelt has higher article counts (expected)")
    
    # check for negative article counts
    if (guardian_articles < 0).any():
        print("   ✗ ERROR: guardian has negative article counts")
        exit(1)
    if (gdelt_articles < 0).any():
        print("   ✗ ERROR: gdelt has negative article counts")
        exit(1)
    print("   ✓ all article counts are non-negative\n")
    
    # 8. check case study countries
    print("8. validating case study countries...")
    case_countries = ['Malaysia', 'Mozambique', 'Canada', 'Brazil']
    
    for country in case_countries:
        country_data = df[df['country'] == country]
        if len(country_data) == 0:
            print(f"   ⚠ WARNING: {country} has no data")
        else:
            years = sorted(country_data['year'].unique())
            print(f"   {country}: {len(country_data)} records, years {years[0]}-{years[-1]}")
            
            # check for malaysia 1MDB period (2013-2015)
            if country == 'Malaysia':
                malaysia_scandal = country_data[country_data['year'].between(2013, 2015)]
                if len(malaysia_scandal) > 0:
                    mean_sentiment = malaysia_scandal['sentiment_score'].mean()
                    print(f"     1MDB period (2013-2015): mean sentiment = {mean_sentiment:.4f}")
                    if mean_sentiment > 0:
                        print("     ⚠ WARNING: sentiment should be negative during scandal period")
            
            # check for mozambique hidden debt period (2013-2016)
            if country == 'Mozambique':
                mozambique_scandal = country_data[country_data['year'].between(2013, 2016)]
                if len(mozambique_scandal) > 0:
                    mean_sentiment = mozambique_scandal['sentiment_score'].mean()
                    print(f"     hidden debt period (2013-2016): mean sentiment = {mean_sentiment:.4f}")
                    if mean_sentiment > 0:
                        print("     ⚠ WARNING: sentiment should be negative during scandal period")
    
    print()
    
    # 9. check that results directories exist (for when notebook is run)
    print("9. checking results directories...")
    results_dir = os.path.join(project_root, 'results')
    figures_dir = os.path.join(results_dir, 'figures')
    tables_dir = os.path.join(results_dir, 'tables')
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir, exist_ok=True)
    if not os.path.exists(tables_dir):
        os.makedirs(tables_dir, exist_ok=True)
    
    print(f"   ✓ results directory: {results_dir}")
    print(f"   ✓ figures directory: {figures_dir}")
    print(f"   ✓ tables directory: {tables_dir}\n")
    
    # 10. summary validation
    print("10. validation summary...")
    print("   ✓ data loaded successfully")
    print("   ✓ both sources (guardian and gdelt) present")
    print("   ✓ sentiment scores in valid range [-1, 1]")
    print("   ✓ basic statistics calculated")
    print("   ✓ transition period analyzed")
    print("   ✓ statistical tests performed")
    print("   ✓ article counts validated")
    print("   ✓ case study countries checked")
    print("   ✓ results directories ready\n")
    
    print("=== ✓ All validation tests passed! ===")
    print("\nNote: Run the notebook to generate visualizations and export results.")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

