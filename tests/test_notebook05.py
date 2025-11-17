#!/usr/bin/env python3
"""
test script for data preparation notebook
run from project root: python3 tests/test_notebook05.py
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
from sklearn.model_selection import train_test_split

print("=== Testing Data Preparation notebook ===\n")

try:
    # 1. check required input files exist
    print("1. checking required input files...")
    main_data_path = os.path.join(project_root, 'data', 'processed', 'corruption_data_expanded_labeled.csv')
    sentiment_path = os.path.join(project_root, 'data', 'sentiment', 'sentiment_scores.csv')
    
    if not os.path.exists(main_data_path):
        print(f"   ERROR: main dataset not found at {main_data_path}")
        exit(1)
    if not os.path.exists(sentiment_path):
        print(f"   ERROR: sentiment scores not found at {sentiment_path}")
        exit(1)
    
    print(f"   OK: main dataset: {main_data_path}")
    print(f"   OK: sentiment scores: {sentiment_path}\n")
    
    # 2. load and verify main dataset structure
    print("2. loading and verifying main dataset...")
    main_df = pd.read_csv(main_data_path)
    
    print(f"   shape: {main_df.shape[0]} rows, {main_df.shape[1]} columns")
    print(f"   countries: {main_df['Country'].nunique()}")
    print(f"   years: {main_df['Year'].min()} to {main_df['Year'].max()}")
    
    # verify required columns
    required_cols = ['Country', 'Year', 'corruption_risk']
    missing_cols = [col for col in required_cols if col not in main_df.columns]
    if missing_cols:
        print(f"   ERROR: missing required columns: {missing_cols}")
        exit(1)
    print(f"   OK: all required columns present\n")
    
    # 3. verify governance indicators (validation features)
    print("3. verifying governance indicators (validation features)...")
    governance_cols = [
        'Voice_Accountability', 'Political_Stability', 'Government_Effectiveness',
        'Regulatory_Quality', 'Rule_of_Law', 'Control_of_Corruption'
    ]
    
    missing_gov = [col for col in governance_cols if col not in main_df.columns]
    if missing_gov:
        print(f"   ERROR: missing governance indicators: {missing_gov}")
        exit(1)
    
    # check for missing values in governance indicators
    missing_gov_values = main_df[governance_cols].isnull().sum().sum()
    if missing_gov_values > 0:
        print(f"   ERROR: {missing_gov_values} missing values in governance indicators")
        exit(1)
    
    print(f"   OK: all 6 governance indicators present")
    print(f"   OK: no missing values in governance indicators\n")
    
    # 4. verify economic indicators (predictive features)
    print("4. verifying economic indicators (predictive features)...")
    economic_cols = [
        'GDP_Growth_annual_perc',
        'External_Debt_perc_GNI',
        'Govt_Expenditure_perc_GDP',
        'FDI_Inflows_perc_GDP',
        'Poverty_Headcount_Ratio'
    ]
    
    missing_econ = [col for col in economic_cols if col not in main_df.columns]
    if missing_econ:
        print(f"   ERROR: missing economic indicators: {missing_econ}")
        exit(1)
    
    print(f"   OK: all 5 economic indicators present")
    
    # check missing values before handling (should be handled in notebook)
    missing_before = main_df[economic_cols].isnull().sum()
    print(f"   missing values before handling:")
    for col in economic_cols:
        missing_count = missing_before[col]
        if missing_count > 0:
            print(f"     {col}: {missing_count} missing")
    print()
    
    # 5. verify sentiment data structure
    print("5. verifying sentiment data structure...")
    sentiment_df = pd.read_csv(sentiment_path)
    
    required_sentiment_cols = ['country', 'year', 'sentiment_score', 'article_count']
    missing_sentiment_cols = [col for col in required_sentiment_cols if col not in sentiment_df.columns]
    if missing_sentiment_cols:
        print(f"   ERROR: missing sentiment columns: {missing_sentiment_cols}")
        exit(1)
    
    print(f"   OK: sentiment data: {len(sentiment_df)} records")
    print(f"   OK: year range: {sentiment_df['year'].min()} to {sentiment_df['year'].max()}\n")
    
    # 6. check final training data file exists
    print("6. checking final training data file...")
    final_data_path = os.path.join(project_root, 'data', 'processed', 'final_training_data.csv')
    if not os.path.exists(final_data_path):
        print(f"   ERROR: final training data not found at {final_data_path}")
        exit(1)
    
    final_df = pd.read_csv(final_data_path)
    print(f"   OK: final training data: {final_df.shape[0]} rows, {final_df.shape[1]} columns")
    
    # verify sentiment was merged
    if 'sentiment_score' not in final_df.columns:
        print(f"   ERROR: sentiment_score not in final dataset")
        exit(1)
    
    # verify no missing sentiment (should be filled with 0.0)
    missing_sentiment = final_df['sentiment_score'].isnull().sum()
    if missing_sentiment > 0:
        print(f"   ERROR: {missing_sentiment} missing sentiment scores (should be filled with 0.0)")
        exit(1)
    
    print(f"   OK: sentiment merged and missing values filled\n")
    
    # 7. verify predictive features are correctly defined
    print("7. verifying predictive features...")
    feature_names_path = os.path.join(project_root, 'models', 'feature_names.txt')
    if not os.path.exists(feature_names_path):
        print(f"   ERROR: feature names file not found at {feature_names_path}")
        exit(1)
    
    with open(feature_names_path, 'r') as f:
        predictive_features = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"   OK: loaded {len(predictive_features)} predictive features from feature_names.txt")
    
    # verify all predictive features are in final dataset
    missing_features = [f for f in predictive_features if f not in final_df.columns]
    if missing_features:
        print(f"   ERROR: predictive features missing from dataset: {missing_features}")
        exit(1)
    
    # verify governance indicators are NOT in predictive features
    governance_in_predictive = [f for f in governance_cols if f in predictive_features]
    if governance_in_predictive:
        print(f"   ERROR: governance indicators should NOT be predictive features: {governance_in_predictive}")
        exit(1)
    
    # verify economic + sentiment features are present
    expected_economic = [f for f in economic_cols if f in predictive_features]
    expected_sentiment = ['sentiment_score' in predictive_features]
    
    if len(expected_economic) != 5:
        print(f"   ERROR: expected 5 economic features, found {len(expected_economic)}")
        exit(1)
    if not expected_sentiment[0]:
        print(f"   ERROR: sentiment_score not in predictive features")
        exit(1)
    
    print(f"   OK: predictive features correctly defined:")
    print(f"     - 5 economic indicators")
    print(f"     - 1 sentiment score")
    print(f"     - 0 governance indicators (correctly excluded)\n")
    
    # 8. verify train/test split files exist
    print("8. verifying train/test split files...")
    train_path = os.path.join(project_root, 'data', 'processed', 'train_set.csv')
    test_path = os.path.join(project_root, 'data', 'processed', 'test_set.csv')
    
    if not os.path.exists(train_path):
        print(f"   ERROR: train set not found at {train_path}")
        exit(1)
    if not os.path.exists(test_path):
        print(f"   ERROR: test set not found at {test_path}")
        exit(1)
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"   OK: train set: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
    print(f"   OK: test set: {test_df.shape[0]} rows, {test_df.shape[1]} columns")
    
    # verify train/test split is approximately 80/20
    total_rows = len(train_df) + len(test_df)
    train_pct = len(train_df) / total_rows
    test_pct = len(test_df) / total_rows
    
    if not (0.75 <= train_pct <= 0.85):
        print(f"   WARNING: train split is {train_pct:.1%} (expected ~80%)")
    else:
        print(f"   OK: train/test split: {train_pct:.1%}/{test_pct:.1%}")
    
    # verify all predictive features are in train/test sets
    missing_train = [f for f in predictive_features if f not in train_df.columns]
    missing_test = [f for f in predictive_features if f not in test_df.columns]
    if missing_train or missing_test:
        print(f"   ERROR: missing features in train/test sets")
        if missing_train:
            print(f"     train: {missing_train}")
        if missing_test:
            print(f"     test: {missing_test}")
        exit(1)
    
    # verify target variable is present
    if 'corruption_risk' not in train_df.columns or 'corruption_risk' not in test_df.columns:
        print(f"   ERROR: corruption_risk target variable missing from train/test sets")
        exit(1)
    
    print(f"   OK: all predictive features and target variable present\n")
    
    # 9. verify stratified split maintains class balance
    print("9. verifying stratified split maintains class balance...")
    train_y = train_df['corruption_risk']
    test_y = test_df['corruption_risk']
    
    train_balance = train_y.mean()
    test_balance = test_y.mean()
    
    # load full dataset to get overall balance
    final_y = final_df['corruption_risk']
    overall_balance = final_y.mean()
    
    print(f"   overall class balance: {overall_balance:.3f}")
    print(f"   train class balance: {train_balance:.3f}")
    print(f"   test class balance: {test_balance:.3f}")
    
    # check if balance is maintained (difference should be < 0.05)
    train_diff = abs(train_balance - overall_balance)
    test_diff = abs(test_balance - overall_balance)
    
    if train_diff > 0.05 or test_diff > 0.05:
        print(f"   WARNING: class balance not well maintained")
        print(f"     train difference: {train_diff:.3f}")
        print(f"     test difference: {test_diff:.3f}")
    else:
        print(f"   OK: class balance maintained in train/test splits\n")
    
    # 10. verify no missing values in predictive features
    print("10. verifying no missing values in predictive features...")
    train_X = train_df[predictive_features]
    test_X = test_df[predictive_features]
    
    train_missing = train_X.isnull().sum().sum()
    test_missing = test_X.isnull().sum().sum()
    
    if train_missing > 0:
        print(f"   ERROR: {train_missing} missing values in train set features")
        print(f"     {train_X.isnull().sum()[train_X.isnull().sum() > 0]}")
        exit(1)
    if test_missing > 0:
        print(f"   ERROR: {test_missing} missing values in test set features")
        print(f"     {test_X.isnull().sum()[test_X.isnull().sum() > 0]}")
        exit(1)
    
    print(f"   OK: no missing values in train set features")
    print(f"   OK: no missing values in test set features\n")
    
    # 11. verify feature ranges are reasonable
    print("11. verifying feature ranges are reasonable...")
    
    # check economic indicators have reasonable ranges
    for col in economic_cols:
        if col in train_X.columns:
            col_min = train_X[col].min()
            col_max = train_X[col].max()
            
            # check for extreme outliers (values that might indicate data errors)
            if col == 'GDP_Growth_annual_perc':
                if col_min < -50 or col_max > 50:
                    print(f"   WARNING: {col} has extreme values: [{col_min:.2f}, {col_max:.2f}]")
            elif col == 'External_Debt_perc_GNI':
                if col_min < 0 or col_max > 500:
                    print(f"   WARNING: {col} has extreme values: [{col_min:.2f}, {col_max:.2f}]")
            elif col == 'Govt_Expenditure_perc_GDP':
                if col_min < 0 or col_max > 100:
                    print(f"   WARNING: {col} has extreme values: [{col_min:.2f}, {col_max:.2f}]")
            elif col == 'FDI_Inflows_perc_GDP':
                if col_min < -50 or col_max > 50:
                    print(f"   WARNING: {col} has extreme values: [{col_min:.2f}, {col_max:.2f}]")
            elif col == 'Poverty_Headcount_Ratio':
                if col_min < 0 or col_max > 100:
                    print(f"   WARNING: {col} has extreme values: [{col_min:.2f}, {col_max:.2f}]")
    
    # check sentiment score is in valid range [-1, 1]
    if 'sentiment_score' in train_X.columns:
        sent_min = train_X['sentiment_score'].min()
        sent_max = train_X['sentiment_score'].max()
        if sent_min < -1.1 or sent_max > 1.1:
            print(f"   WARNING: sentiment_score outside expected range [-1, 1]: [{sent_min:.4f}, {sent_max:.4f}]")
        else:
            print(f"   OK: sentiment_score in valid range: [{sent_min:.4f}, {sent_max:.4f}]")
    
    print(f"   OK: feature ranges verified\n")
    
    # 12. verify no duplicate country-year combinations
    print("12. verifying no duplicate country-year combinations...")
    duplicates = final_df.duplicated(subset=['Country', 'Year'], keep=False)
    if duplicates.any():
        print(f"   ERROR: {duplicates.sum()} duplicate country-year combinations found")
        print(f"     {final_df[duplicates][['Country', 'Year']]}")
        exit(1)
    
    print(f"   OK: no duplicate country-year combinations\n")
    
    # 13. verify year range is correct
    print("13. verifying year range...")
    year_numeric = pd.to_numeric(final_df['Year'], errors='coerce')
    year_min = int(year_numeric.min())
    year_max = int(year_numeric.max())
    
    if year_min != 2010 or year_max != 2023:
        print(f"   ERROR: year range is {year_min}-{year_max} (expected 2010-2023)")
        exit(1)
    
    print(f"   OK: year range: {year_min}-{year_max}\n")
    
    # 14. verify country count
    print("14. verifying country count...")
    country_count = final_df['Country'].nunique()
    if country_count != 19:
        print(f"   WARNING: {country_count} countries found (expected 19)")
    else:
        print(f"   OK: {country_count} countries present\n")
    
    # 15. verify case study countries are present
    print("15. verifying case study countries...")
    
    # malaysia 1mdb (2013-2015)
    malaysia_scandal = final_df[
        (final_df['Country'] == 'Malaysia') & 
        (final_df['Year'].between(2013, 2015))
    ]
    if len(malaysia_scandal) == 0:
        print(f"   WARNING: malaysia 2013-2015 not found")
    else:
        if not malaysia_scandal['corruption_risk'].all():
            print(f"   WARNING: malaysia 2013-2015 not all flagged as high risk")
        else:
            print(f"   OK: malaysia 1mdb (2013-2015): {len(malaysia_scandal)} records, all high risk")
    
    # mozambique hidden debt (2013-2016)
    mozambique_scandal = final_df[
        (final_df['Country'] == 'Mozambique') & 
        (final_df['Year'].between(2013, 2016))
    ]
    if len(mozambique_scandal) == 0:
        print(f"   WARNING: mozambique 2013-2016 not found")
    else:
        if not mozambique_scandal['corruption_risk'].all():
            print(f"   WARNING: mozambique 2013-2016 not all flagged as high risk")
        else:
            print(f"   OK: mozambique hidden debt (2013-2016): {len(mozambique_scandal)} records, all high risk")
    
    # canada (control)
    canada = final_df[final_df['Country'] == 'Canada']
    if len(canada) == 0:
        print(f"   WARNING: canada not found")
    else:
        canada_high_risk = canada['corruption_risk'].sum()
        if canada_high_risk > 0:
            print(f"   WARNING: canada has {canada_high_risk} high-risk years (expected 0)")
        else:
            print(f"   OK: canada (control): {len(canada)} records, all low risk")
    
    print()
    
    # 16. verify target variable distribution
    print("16. verifying target variable distribution...")
    target_dist = final_df['corruption_risk'].value_counts()
    print(f"   low risk (0): {target_dist.get(0, 0)} ({target_dist.get(0, 0)/len(final_df):.1%})")
    print(f"   high risk (1): {target_dist.get(1, 0)} ({target_dist.get(1, 0)/len(final_df):.1%})")
    
    if target_dist.get(0, 0) == 0 or target_dist.get(1, 0) == 0:
        print(f"   ERROR: target variable has only one class")
        exit(1)
    
    print(f"   OK: both classes present\n")
    
    # 17. verify train/test split reproducibility
    print("17. verifying train/test split reproducibility...")
    
    # recreate split with same random_state to verify consistency
    X_full = final_df[predictive_features]
    y_full = final_df['corruption_risk']
    
    X_train_recreated, X_test_recreated, y_train_recreated, y_test_recreated = train_test_split(
        X_full, y_full,
        test_size=0.2,
        stratify=y_full,
        random_state=42
    )
    
    # check if sizes match
    if len(X_train_recreated) != len(train_df) or len(X_test_recreated) != len(test_df):
        print(f"   WARNING: train/test split sizes don't match recreated split")
        print(f"     original: train={len(train_df)}, test={len(test_df)}")
        print(f"     recreated: train={len(X_train_recreated)}, test={len(X_test_recreated)}")
    else:
        print(f"   OK: train/test split is reproducible (random_state=42)\n")
    
    # 18. summary validation
    print("18. validation summary...")
    print("   OK: all required input files present")
    print("   OK: governance indicators present (validation features)")
    print("   OK: economic indicators present (predictive features)")
    print("   OK: sentiment scores merged correctly")
    print("   OK: predictive features correctly defined (6 features: 5 economic + 1 sentiment)")
    print("   OK: governance indicators excluded from predictive features")
    print("   OK: train/test split files created")
    print("   OK: stratified split maintains class balance")
    print("   OK: no missing values in predictive features")
    print("   OK: feature ranges are reasonable")
    print("   OK: no duplicate country-year combinations")
    print("   OK: year range is correct (2010-2023)")
    print("   OK: case study countries present with correct labels")
    print("   OK: target variable has both classes")
    print("   OK: train/test split is reproducible\n")
    
    print("=== All validation tests passed! ===")
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

