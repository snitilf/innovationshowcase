#!/usr/bin/env python3
"""
regenerate train/test sets with only predictive features (economic + sentiment).
this fixes the circular reasoning issue by removing governance indicators from features.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# set working directory to project root
current_dir = os.getcwd()
if current_dir.endswith('scripts'):
    os.chdir('..')
elif 'scripts' in current_dir:
    project_root = current_dir.split('scripts')[0].rstrip('/')
    if os.path.exists(project_root):
        os.chdir(project_root)

print(f"working directory: {os.getcwd()}")

# load final training data
final_df = pd.read_csv('data/processed/final_training_data.csv')
print(f"\nloaded final training data: {final_df.shape}")

# define predictive features (economic + sentiment only)
predictive_features = [
    'GDP_Growth_annual_perc',
    'External_Debt_perc_GNI',
    'Govt_Expenditure_perc_GDP',
    'FDI_Inflows_perc_GDP',
    'Poverty_Headcount_Ratio',
    'sentiment_score'
]

print(f"\npredictive features ({len(predictive_features)}):")
for i, feature in enumerate(predictive_features, 1):
    print(f"  {i}. {feature}")

# extract feature matrix (only predictive features)
X = final_df[predictive_features].copy()
y = final_df['corruption_risk'].copy()

print(f"\nfeature matrix shape: {X.shape}")
print(f"target vector shape: {y.shape}")

# verify no missing values
missing = X.isnull().sum().sum()
if missing > 0:
    print(f"\n⚠️  warning: {missing} missing values in feature matrix")
    print(X.isnull().sum())
else:
    print("\n✓ no missing values in feature matrix")

# create stratified train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(f"\ntrain set shape: {X_train.shape}")
print(f"test set shape: {X_test.shape}")

print(f"\nclass distribution (train):")
print(y_train.value_counts())
print(f"class distribution (%):")
print(y_train.value_counts(normalize=True))

print(f"\nclass distribution (test):")
print(y_test.value_counts())
print(f"class distribution (%):")
print(y_test.value_counts(normalize=True))

# save train and test sets
os.makedirs('data/processed', exist_ok=True)

train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

train_df.to_csv('data/processed/train_set.csv', index=False)
test_df.to_csv('data/processed/test_set.csv', index=False)

print(f"\n✓ saved train_set.csv: {train_df.shape}")
print(f"✓ saved test_set.csv: {test_df.shape}")
print(f"\n✓ train/test sets regenerated with only predictive features")
print(f"  (governance indicators removed to avoid circular reasoning)")

