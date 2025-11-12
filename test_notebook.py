#!/usr/bin/env python3
"""
test script for world bank API notebook
run from project root: python3 test_notebook.py
"""

import wbdata
import pandas as pd
import datetime
import os

print("=== Testing World Bank API notebook ===\n")

# country codes as used by world bank
countries = ["CAN", "MYS", "MOZ"]

# date range for historical data
data_range = (datetime.datetime(2010, 1, 1), datetime.datetime(2024, 1, 1))

# defining all indicators
indicators = {
    'VA.EST': 'Voice_Accountability',
    'PV.EST': 'Political_Stability',
    'GE.EST': 'Government_Effectiveness',
    'RQ.EST': 'Regulatory_Quality',
    'RL.EST': 'Rule_of_Law',
    'CC.EST': 'Control_of_Corruption',
    'DT.DOD.DECT.GN.ZS': 'External_Debt_perc_GNI',
    'NY.GDP.MKTP.KD.ZG': 'GDP_Growth_annual_perc',
    'GC.XPN.TOTL.GD.ZS': 'Govt_Expenditure_perc_GDP',
    'BX.KLT.DINV.WD.GD.ZS': 'FDI_Inflows_perc_GDP',
    'SI.POV.DDAY': 'Poverty_Headcount_Ratio'
}

try:
    print("1. fetching data from API...")
    df = wbdata.get_dataframe(indicators, 
                              country=countries, 
                              date=data_range,
                              parse_dates=False)
    print("   ✓ data fetched successfully\n")
    
    # data processing
    df = df.reset_index()
    df = df.rename(columns={'date': 'Year', 'country': 'Country'})
    column_order = ['Country', 'Year'] + list(indicators.values())
    existing_columns = [col for col in column_order if col in df.columns]
    df = df[existing_columns]
    df = df.sort_values(by=['Country', 'Year']).reset_index(drop=True)
    
    print(f"2. data shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   years: {df['Year'].min()} to {df['Year'].max()}\n")
    
    # save test
    os.makedirs('data/raw', exist_ok=True)
    output_path = 'data/raw/corruption_data_baseline.csv'
    df.to_csv(output_path, index=False)
    
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"3. ✓ file saved: {output_path} ({file_size} bytes)\n")
    else:
        print(f"3. ✗ ERROR: file not created\n")
        exit(1)
    
    # governance check
    print("4. sample governance scores (Canada 2023):")
    canada_2023 = df[(df['Country'] == 'Canada') & (df['Year'] == '2023')]
    if not canada_2023.empty:
        print(f"   Control_of_Corruption: {canada_2023['Control_of_Corruption'].values[0]:.2f}")
        print(f"   Rule_of_Law: {canada_2023['Rule_of_Law'].values[0]:.2f}")
    
    print("\n=== ✓ All tests passed! ===")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)