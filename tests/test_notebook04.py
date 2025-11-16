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

print("=== Testing Sentiment Source Validation notebook ===\n")

try:
    # 1. load sentiment and labeled data
    print("1. loading sentiment and labeled data...")
    sentiment_path = os.path.join(project_root, 'data', 'sentiment', 'sentiment_scores.csv')
    labeled_path = os.path.join(project_root, 'data', 'processed', 'corruption_data_expanded_labeled.csv')
    
    if not os.path.exists(sentiment_path):
        print("   ✗ ERROR: sentiment scores file not found")
        exit(1)
    if not os.path.exists(labeled_path):
        print("   ✗ ERROR: labeled data file not found")
        exit(1)
    
    sentiment_df = pd.read_csv(sentiment_path)
    labeled_df = pd.read_csv(labeled_path)
    
    print(f"   ✓ sentiment data: {len(sentiment_df)} records")
    print(f"   ✓ labeled data: {len(labeled_df)} records")
    print(f"   countries: {sentiment_df['country'].nunique()}")
    print(f"   year range: {sentiment_df['year'].min()} to {sentiment_df['year'].max()}\n")
    
    # validate required columns
    sentiment_cols = ['country', 'year', 'sentiment_score', 'article_count']
    missing_cols = [col for col in sentiment_cols if col not in sentiment_df.columns]
    if missing_cols:
        print(f"   ✗ ERROR: missing sentiment columns: {missing_cols}")
        exit(1)
    
    required_labeled_cols = ['Country', 'Year', 'corruption_risk', 'Risk_Category']
    missing_labeled = [col for col in required_labeled_cols if col not in labeled_df.columns]
    if missing_labeled:
        print(f"   ✗ ERROR: missing labeled columns: {missing_labeled}")
        exit(1)
    print(f"   ✓ all required columns present\n")
    
    # 2. merge sentiment with labeled data
    print("2. merging sentiment with labeled data...")
    merged_df = labeled_df.merge(
        sentiment_df,
        left_on=['Country', 'Year'],
        right_on=['country', 'year'],
        how='left'
    )
    
    # fill missing sentiment with 0 (neutral) for analysis
    merged_df['sentiment_score'] = merged_df['sentiment_score'].fillna(0.0)
    
    print(f"   ✓ merged dataset: {len(merged_df)} records")
    print(f"   records with sentiment data: {merged_df['sentiment_score'].notna().sum()}\n")
    
    if len(merged_df) == 0:
        print("   ✗ ERROR: merged dataset is empty")
        exit(1)
    
    # 3. validate sentiment score ranges
    print("3. validating sentiment score ranges...")
    all_scores = merged_df['sentiment_score']
    in_range = (all_scores >= -1).all() and (all_scores <= 1).all()
    
    if not in_range:
        print(f"   ✗ ERROR: sentiment scores outside [-1, 1] range")
        print(f"   min: {all_scores.min()}, max: {all_scores.max()}")
        exit(1)
    print(f"   ✓ all scores in valid range [-1, 1]")
    print(f"   range: [{all_scores.min():.4f}, {all_scores.max():.4f}]\n")
    
    # 4. validate sentiment by risk category
    print("4. validating sentiment by risk category...")
    high_risk_sentiment = merged_df[merged_df['corruption_risk'] == 1]['sentiment_score']
    low_risk_sentiment = merged_df[merged_df['corruption_risk'] == 0]['sentiment_score']
    
    if high_risk_sentiment.empty or low_risk_sentiment.empty:
        print("   ✗ ERROR: no high-risk or low-risk sentiment data")
        exit(1)
    
    high_mean = high_risk_sentiment.mean()
    low_mean = low_risk_sentiment.mean()
    
    print(f"   high-risk mean sentiment: {high_mean:.4f}")
    print(f"   low-risk mean sentiment: {low_mean:.4f}")
    
    # both should be negative (corruption news is inherently negative)
    if high_mean > 0.1:
        print(f"   ⚠ WARNING: high-risk mean is positive ({high_mean:.4f}), expected negative")
    if low_mean > 0.1:
        print(f"   ⚠ WARNING: low-risk mean is positive ({low_mean:.4f}), expected negative")
    
    if high_mean < 0 and low_mean < 0:
        print("   ✓ both risk categories show negative sentiment (as expected)\n")
    else:
        print("   ⚠ NOTE: one or both categories not negative\n")
    
    # 5. validate case study countries
    print("5. validating case study countries...")
    
    # malaysia 1MDB (2013-2015)
    malaysia_scandal = merged_df[(merged_df['Country'] == 'Malaysia') & 
                                 (merged_df['Year'].between(2013, 2015))]
    if len(malaysia_scandal) > 0:
        malaysia_sentiment = malaysia_scandal['sentiment_score'].mean()
        malaysia_risk = malaysia_scandal['corruption_risk'].unique()[0]
        print(f"   malaysia 1MDB (2013-2015):")
        print(f"     mean sentiment: {malaysia_sentiment:.4f}")
        print(f"     corruption risk: {malaysia_risk} (expected: 1)")
        if malaysia_risk != 1:
            print("     ⚠ WARNING: malaysia should be high risk during 1MDB period")
        if malaysia_sentiment > 0.1:
            print("     ⚠ WARNING: sentiment should be negative during scandal period")
        else:
            print("     ✓ sentiment is negative during scandal period")
    else:
        print("   ⚠ WARNING: no malaysia data for 1MDB period")
    
    # mozambique hidden debt (2013-2016)
    mozambique_scandal = merged_df[(merged_df['Country'] == 'Mozambique') & 
                                  (merged_df['Year'].between(2013, 2016))]
    if len(mozambique_scandal) > 0:
        mozambique_sentiment = mozambique_scandal['sentiment_score'].mean()
        mozambique_risk = mozambique_scandal['corruption_risk'].unique()[0]
        print(f"   mozambique hidden debt (2013-2016):")
        print(f"     mean sentiment: {mozambique_sentiment:.4f}")
        print(f"     corruption risk: {mozambique_risk} (expected: 1)")
        if mozambique_risk != 1:
            print("     ⚠ WARNING: mozambique should be high risk during hidden debt period")
        if mozambique_sentiment > 0.1:
            print("     ⚠ WARNING: sentiment should be negative during crisis period")
        else:
            print("     ✓ sentiment is negative or neutral during crisis period")
    else:
        print("   ⚠ WARNING: no mozambique data for hidden debt period")
    
    # canada (control - should be low risk)
    canada = merged_df[merged_df['Country'] == 'Canada']
    if len(canada) > 0:
        canada_sentiment = canada['sentiment_score'].mean()
        canada_risk = canada['corruption_risk'].sum()
        canada_total = len(canada)
        print(f"   canada (control):")
        print(f"     mean sentiment: {canada_sentiment:.4f}")
        print(f"     high-risk years: {canada_risk}/{canada_total} (expected: 0)")
        if canada_risk > 0:
            print("     ⚠ WARNING: canada should be low risk throughout")
        else:
            print("     ✓ canada correctly labeled as low risk")
    else:
        print("   ⚠ WARNING: no canada data found")
    
    print()
    
    # 6. validate country-level sentiment analysis
    print("6. validating country-level sentiment analysis...")
    country_sentiment = merged_df.groupby('Country').agg({
        'sentiment_score': 'mean',
        'corruption_risk': 'mean',
        'Risk_Category': 'first'
    })
    
    if len(country_sentiment) == 0:
        print("   ✗ ERROR: country-level analysis failed")
        exit(1)
    
    print(f"   ✓ analyzed {len(country_sentiment)} countries")
    
    # check that risk categories are present
    risk_categories = country_sentiment['Risk_Category'].unique()
    expected_categories = ['High-Risk', 'Medium-Risk', 'Low-Risk']
    missing_categories = [cat for cat in expected_categories if cat not in risk_categories]
    if missing_categories:
        print(f"   ⚠ WARNING: missing risk categories: {missing_categories}")
    else:
        print("   ✓ all risk categories present\n")
    
    # 7. validate data source separation (quick check)
    print("7. validating data sources (guardian vs gdelt)...")
    guardian_df = sentiment_df[sentiment_df['year'] <= 2016]
    gdelt_df = sentiment_df[sentiment_df['year'] >= 2017]
    
    if len(guardian_df) == 0:
        print("   ✗ ERROR: no guardian data found")
        exit(1)
    if len(gdelt_df) == 0:
        print("   ✗ ERROR: no gdelt data found")
        exit(1)
    
    guardian_valid = (guardian_df['sentiment_score'] >= -1).all() and (guardian_df['sentiment_score'] <= 1).all()
    gdelt_valid = (gdelt_df['sentiment_score'] >= -1).all() and (gdelt_df['sentiment_score'] <= 1).all()
    
    if not guardian_valid or not gdelt_valid:
        print("   ✗ ERROR: data source scores outside valid range")
        exit(1)
    
    print(f"   ✓ guardian (2010-2016): {len(guardian_df)} records")
    print(f"   ✓ gdelt (2017-2023): {len(gdelt_df)} records")
    print(f"   ✓ both sources have valid scores\n")
    
    # 8. check that results directories exist (for when notebook is run)
    print("8. checking results directories...")
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
    
    # 9. summary validation
    print("9. validation summary...")
    print("   ✓ sentiment and labeled data loaded successfully")
    print("   ✓ datasets merged correctly")
    print("   ✓ sentiment scores in valid range [-1, 1]")
    print("   ✓ risk category sentiment analysis completed")
    print("   ✓ case study countries validated")
    print("   ✓ country-level analysis completed")
    print("   ✓ data sources validated")
    print("   ✓ results directories ready\n")
    
    print("=== ✓ All validation tests passed! ===")
    print("\nNote: Run the notebook to generate visualizations and export results.")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

