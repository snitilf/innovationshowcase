#!/usr/bin/env python3
"""
test script for data cleaning and labeling notebook
run from project root: python3 tests/test_notebook02.py
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

print("=== Testing Data Cleaning & Labeling notebook ===\n")

try:
    # 1. load baseline data
    print("1. loading baseline data...")
    baseline_path = os.path.join(project_root, 'data', 'raw', 'corruption_data_baseline.csv')
    if not os.path.exists(baseline_path):
        print("   ✗ ERROR: baseline data file not found")
        exit(1)
    
    df = pd.read_csv(baseline_path)
    print(f"   ✓ loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   years: {df['Year'].min()} to {df['Year'].max()}\n")
    
    # 2. drop 2024
    print("2. dropping 2024 data...")
    rows_before = len(df)
    df = df[df['Year'] != 2024].copy()
    rows_after = len(df)
    print(f"   rows: {rows_before} → {rows_after}")
    print(f"   years: {df['Year'].min()} to {df['Year'].max()}\n")
    
    # 3. handle missing values
    print("3. handling missing values...")
    governance_cols = ['Voice_Accountability', 'Political_Stability', 'Government_Effectiveness', 
                       'Regulatory_Quality', 'Rule_of_Law', 'Control_of_Corruption']
    economic_cols = ['External_Debt_perc_GNI', 'GDP_Growth_annual_perc', 
                     'Govt_Expenditure_perc_GDP', 'FDI_Inflows_perc_GDP', 'Poverty_Headcount_Ratio']
    
    # check governance indicators (should have none missing after dropping 2024)
    missing_governance = df[df[governance_cols].isnull().any(axis=1)]
    if len(missing_governance) > 0:
        print(f"   ✗ ERROR: {len(missing_governance)} rows with missing governance indicators")
        exit(1)
    print(f"   ✓ no missing governance indicators")
    
    # forward fill economic indicators
    for col in economic_cols:
        df[col] = df.groupby('Country')[col].ffill()
    
    print(f"   ✓ forward filled economic indicators\n")
    
    # 4. create corruption risk labels
    print("4. creating corruption risk labels...")
    thresholds = {
        'Voice_Accountability': 1.15,
        'Political_Stability': 0.50,
        'Government_Effectiveness': 1.15,
        'Regulatory_Quality': 1.15,
        'Rule_of_Law': 1.15,
        'Control_of_Corruption': 1.15
    }
    
    # create flag columns
    for indicator, threshold in thresholds.items():
        flag_col = f'{indicator}_flag'
        df[flag_col] = (df[indicator] < threshold).astype(int)
    
    flag_cols = [f'{ind}_flag' for ind in thresholds.keys()]
    df['total_flags'] = df[flag_cols].sum(axis=1)
    df['corruption_risk'] = (df['total_flags'] >= 4).astype(int)
    
    print(f"   flag distribution: {dict(df['total_flags'].value_counts().sort_index())}")
    print(f"   risk labels: {dict(df['corruption_risk'].value_counts())}\n")
    
    # 5. validate against known scandals
    print("5. validating labels against known scandals...")
    
    # malaysia 1MDB (2013-2015)
    malaysia_scandal = df[(df['Country'] == 'Malaysia') & (df['Year'].between(2013, 2015))]
    if not malaysia_scandal['corruption_risk'].all():
        print(f"   ✗ ERROR: Malaysia 2013-2015 not all flagged as high risk")
        exit(1)
    print(f"   ✓ Malaysia 2013-2015: all flagged as high risk")
    
    # mozambique hidden debt (2013-2016)
    mozambique_scandal = df[(df['Country'] == 'Mozambique') & (df['Year'].between(2013, 2016))]
    if not mozambique_scandal['corruption_risk'].all():
        print(f"   ✗ ERROR: Mozambique 2013-2016 not all flagged as high risk")
        exit(1)
    print(f"   ✓ Mozambique 2013-2016: all flagged as high risk")
    
    # canada (control - should be low risk)
    canada = df[df['Country'] == 'Canada']
    canada_high_risk = canada['corruption_risk'].sum()
    if canada_high_risk > 0:
        print(f"   ✗ ERROR: Canada has {canada_high_risk} high risk years (expected 0)")
        exit(1)
    print(f"   ✓ Canada: {len(canada)} years, all low risk\n")
    
    # 6. save processed dataset
    print("6. saving processed dataset...")
    processed_dir = os.path.join(project_root, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    output_path = os.path.join(processed_dir, 'corruption_data_labeled.csv')
    df.to_csv(output_path, index=False)
    
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"   ✓ file saved: {output_path} ({file_size} bytes)")
        print(f"   shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
    else:
        print(f"   ✗ ERROR: file not created\n")
        exit(1)
    
    # 7. verify expected columns
    print("7. verifying dataset structure...")
    expected_cols = ['Country', 'Year', 'corruption_risk', 'total_flags'] + list(thresholds.keys())
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        print(f"   ✗ ERROR: missing columns: {missing_cols}")
        exit(1)
    print(f"   ✓ all expected columns present")
    print(f"   ✓ total columns: {len(df.columns)}\n")
    
    print("=== ✓ All tests passed! ===")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

