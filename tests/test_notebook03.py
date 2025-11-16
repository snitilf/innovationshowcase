#!/usr/bin/env python3
"""
test script for expanded dataset notebook
run from project root: python3 tests/test_notebook03.py
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

print("=== Testing Expanded Dataset notebook ===\n")

# expected countries
# note: World Bank API may return country names with suffixes (e.g., "Venezuela, RB")
baseline_countries = ['Canada', 'Malaysia', 'Mozambique']
high_risk_countries = ['Angola', 'Venezuela', 'Zimbabwe', 'Iraq', 'Ukraine']
medium_risk_countries = ['Brazil', 'South Africa', 'India', 'Philippines']
low_risk_countries = ['Norway', 'Denmark', 'Singapore', 'Australia', 'New Zealand', 'Switzerland', 'Germany']

all_expected_countries = baseline_countries + high_risk_countries + medium_risk_countries + low_risk_countries

# world bank country name variations (with suffixes)
wb_country_variations = {
    'Venezuela': ['Venezuela', 'Venezuela, RB', 'Venezuela, Bolivarian Republic of']
}

try:
    # 1. check expanded dataset exists
    print("1. checking expanded dataset file...")
    expanded_path = os.path.join(project_root, 'data', 'raw', 'corruption_data_expanded.csv')
    if not os.path.exists(expanded_path):
        print(f"   ✗ ERROR: expanded dataset not found at {expanded_path}")
        exit(1)
    print(f"   ✓ file exists: {expanded_path}\n")
    
    # 2. load expanded dataset
    print("2. loading expanded dataset...")
    df_expanded = pd.read_csv(expanded_path)
    print(f"   ✓ loaded: {df_expanded.shape[0]} rows, {df_expanded.shape[1]} columns")
    print(f"   years: {df_expanded['Year'].min()} to {df_expanded['Year'].max()}\n")
    
    # 3. verify no 2024 data
    print("3. verifying no 2024 data...")
    # check both string and numeric 2024
    has_2024 = (df_expanded['Year'] == '2024').any() or (df_expanded['Year'] == 2024).any()
    if has_2024:
        print(f"   ✗ ERROR: dataset contains 2024 data (should be excluded)")
        exit(1)
    print(f"   ✓ no 2024 data present\n")
    
    # 4. check countries present
    print("4. checking countries in dataset...")
    countries_in_data = sorted(df_expanded['Country'].unique())
    print(f"   countries found: {len(countries_in_data)}")
    
    # check all expected countries are present (accounting for World Bank name variations)
    missing_countries = []
    for country in all_expected_countries:
        # check if country name exists as-is or in variations
        found = False
        if country in countries_in_data:
            found = True
        else:
            # check for World Bank variations
            variations = wb_country_variations.get(country, [])
            for variation in variations:
                if variation in countries_in_data:
                    found = True
                    print(f"   ℹ {country} found as '{variation}' in dataset")
                    break
        
        if not found:
            missing_countries.append(country)
    
    if missing_countries:
        print(f"   ✗ ERROR: missing countries: {missing_countries}")
        print(f"   available countries: {countries_in_data}")
        exit(1)
    print(f"   ✓ all expected countries present")
    print(f"   countries: {countries_in_data}\n")
    
    # 5. verify data structure matches baseline
    print("5. verifying data structure...")
    expected_cols = ['Country', 'Year', 'Voice_Accountability', 'Political_Stability', 
                     'Government_Effectiveness', 'Regulatory_Quality', 'Rule_of_Law', 
                     'Control_of_Corruption']
    missing_cols = [col for col in expected_cols if col not in df_expanded.columns]
    if missing_cols:
        print(f"   ✗ ERROR: missing columns: {missing_cols}")
        exit(1)
    print(f"   ✓ all expected columns present\n")
    
    # 6. check data completeness by country
    print("6. checking data completeness by country...")
    country_counts = df_expanded['Country'].value_counts().sort_index()
    expected_rows_per_country = 14  # 2010-2023
    
    incomplete_countries = []
    for country, count in country_counts.items():
        if count < expected_rows_per_country:
            incomplete_countries.append((country, count))
    
    if incomplete_countries:
        print(f"   ⚠ WARNING: some countries have incomplete data:")
        for country, count in incomplete_countries:
            print(f"      {country}: {count} rows (expected {expected_rows_per_country})")
    else:
        print(f"   ✓ all countries have complete data (14 rows each)")
    print(f"   total rows: {df_expanded.shape[0]} (expected: ~{len(all_expected_countries) * expected_rows_per_country})\n")
    
    # 7. check governance indicators quality
    print("7. checking governance indicators quality...")
    governance_cols = ['Voice_Accountability', 'Political_Stability', 'Government_Effectiveness', 
                       'Regulatory_Quality', 'Rule_of_Law', 'Control_of_Corruption']
    
    # check for missing governance data
    missing_gov = df_expanded[df_expanded[governance_cols].isnull().any(axis=1)]
    if len(missing_gov) > 0:
        print(f"   ⚠ WARNING: {len(missing_gov)} rows with missing governance indicators")
        print(f"      countries affected: {missing_gov['Country'].unique()}")
    else:
        print(f"   ✓ no missing governance indicators")
    
    # check governance scores are reasonable (should be between -2.5 and 2.5 typically)
    for col in governance_cols:
        if df_expanded[col].min() < -3 or df_expanded[col].max() > 3:
            print(f"   ⚠ WARNING: {col} has unusual values (min: {df_expanded[col].min():.2f}, max: {df_expanded[col].max():.2f})")
    print()
    
    # 8. verify risk categories make sense (sample check)
    print("8. verifying governance scores align with risk categories...")
    
    # check a high-risk country (should have low governance scores)
    if 'Angola' in countries_in_data:
        angola_avg = df_expanded[df_expanded['Country'] == 'Angola'][governance_cols].mean()
        if angola_avg['Control_of_Corruption'].mean() > 0:
            print(f"   ⚠ WARNING: Angola (high-risk) has positive corruption control score")
        else:
            print(f"   ✓ Angola (high-risk): Control_of_Corruption = {angola_avg['Control_of_Corruption']:.2f}")
    
    # check a low-risk country (should have high governance scores)
    if 'Norway' in countries_in_data:
        norway_avg = df_expanded[df_expanded['Country'] == 'Norway'][governance_cols].mean()
        if norway_avg['Control_of_Corruption'].mean() < 1.0:
            print(f"   ⚠ WARNING: Norway (low-risk) has low corruption control score")
        else:
            print(f"   ✓ Norway (low-risk): Control_of_Corruption = {norway_avg['Control_of_Corruption']:.2f}")
    
    # check baseline countries still present
    for country in baseline_countries:
        if country in countries_in_data:
            country_data = df_expanded[df_expanded['Country'] == country]
            print(f"   ✓ {country}: {len(country_data)} rows")
    print()
    
    # 9. compare with baseline dataset
    print("9. comparing with baseline dataset...")
    baseline_path = os.path.join(project_root, 'data', 'raw', 'corruption_data_baseline.csv')
    if os.path.exists(baseline_path):
        df_baseline = pd.read_csv(baseline_path)
        baseline_rows = len(df_baseline[df_baseline['Year'] != '2024'])
        expanded_rows = len(df_expanded)
        
        print(f"   baseline (excluding 2024): {baseline_rows} rows")
        print(f"   expanded: {expanded_rows} rows")
        print(f"   additional rows: {expanded_rows - baseline_rows}")
        
        # verify baseline countries match
        for country in baseline_countries:
            baseline_count = len(df_baseline[(df_baseline['Country'] == country) & (df_baseline['Year'] != '2024')])
            expanded_count = len(df_expanded[df_expanded['Country'] == country])
            if baseline_count != expanded_count:
                print(f"   ⚠ WARNING: {country} row count mismatch (baseline: {baseline_count}, expanded: {expanded_count})")
            else:
                print(f"   ✓ {country}: consistent row count ({baseline_count})")
    else:
        print(f"   ⚠ baseline file not found, skipping comparison")
    print()
    
    # 10. summary statistics
    print("10. summary statistics...")
    print(f"   total countries: {df_expanded['Country'].nunique()}")
    print(f"   total rows: {df_expanded.shape[0]}")
    print(f"   years covered: {df_expanded['Year'].min()} to {df_expanded['Year'].max()}")
    print(f"   columns: {df_expanded.shape[1]}")
    
    # governance score ranges
    print(f"\n   governance score ranges:")
    for col in governance_cols:
        print(f"      {col}: {df_expanded[col].min():.2f} to {df_expanded[col].max():.2f}")
    print()
    
    print("=== ✓ All tests passed! ===")
    print(f"\nExpanded dataset is ready for Step 2 (labeling process)")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

