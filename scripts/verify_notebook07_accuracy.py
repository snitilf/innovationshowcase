#!/usr/bin/env python3
"""
verify all calculations and numbers in notebook 07 are accurate
"""

import os
import sys

# get project root directory (parent of scripts/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# automatically use venv python if it exists and we're not already using it
venv_python = os.path.join(project_root, 'venv', 'bin', 'python3')
if os.path.exists(venv_python) and not sys.executable.startswith(os.path.join(project_root, 'venv')):
    # re-execute this script with venv python
    os.execv(venv_python, [venv_python] + sys.argv)

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import joblib

# change to project root
os.chdir(project_root)

print("="*70)
print("VERIFYING NOTEBOOK 07 CALCULATIONS")
print("="*70)

# load model and data
dt_model = joblib.load('models/decision_tree_model.pkl')
test_df = pd.read_csv('data/processed/test_set.csv')

with open('models/feature_names.txt', 'r') as f:
    feature_names = [line.strip() for line in f.readlines()]

X_test = test_df[feature_names]
y_test = test_df['corruption_risk']

# generate predictions
y_pred = dt_model.predict(X_test)
y_pred_proba = dt_model.predict_proba(X_test)[:, 1]

# calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

print("\n1. VERIFYING CONFUSION MATRIX")
print("-" * 70)
print(f"Confusion Matrix values:")
print(f"  TN (True Negatives):  {tn}")
print(f"  FP (False Positives): {fp}")
print(f"  FN (False Negatives): {fn}")
print(f"  TP (True Positives):  {tp}")
print(f"  Total: {tn + fp + fn + tp} (should be {len(y_test)})")

if (tn + fp + fn + tp) != len(y_test):
    print(f"  ERROR: Total doesn't match test set size!")
else:
    print(f"  OK: Total matches test set size")

# verify confusion matrix matches actual counts
actual_low = (y_test == 0).sum()
actual_high = (y_test == 1).sum()
pred_low = (y_pred == 0).sum()
pred_high = (y_pred == 1).sum()

print(f"\nActual distribution: Low={actual_low}, High={actual_high}")
print(f"Predicted distribution: Low={pred_low}, High={pred_high}")

# verify confusion matrix breakdown
print(f"\nVerifying confusion matrix breakdown:")
print(f"  Actual Low: TN={tn} + FP={fp} = {tn+fp} (should be {actual_low})")
print(f"  Actual High: FN={fn} + TP={tp} = {fn+tp} (should be {actual_high})")
print(f"  Predicted Low: TN={tn} + FN={fn} = {tn+fn} (should be {pred_low})")
print(f"  Predicted High: FP={fp} + TP={tp} = {fp+tp} (should be {pred_high})")

errors = []
if (tn + fp) != actual_low:
    errors.append(f"Actual Low mismatch: {tn+fp} vs {actual_low}")
if (fn + tp) != actual_high:
    errors.append(f"Actual High mismatch: {fn+tp} vs {actual_high}")
if (tn + fn) != pred_low:
    errors.append(f"Predicted Low mismatch: {tn+fn} vs {pred_low}")
if (fp + tp) != pred_high:
    errors.append(f"Predicted High mismatch: {fp+tp} vs {pred_high}")

if errors:
    print("  ERRORS FOUND:")
    for e in errors:
        print(f"    - {e}")
else:
    print("  OK: All confusion matrix breakdowns are correct")

print("\n2. VERIFYING METRICS FROM CONFUSION MATRIX")
print("-" * 70)

# manual calculations
manual_accuracy = (tp + tn) / (tp + tn + fp + fn)
manual_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
manual_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
manual_f1 = 2 * (manual_precision * manual_recall) / (manual_precision + manual_recall) if (manual_precision + manual_recall) > 0 else 0

print(f"Accuracy:")
print(f"  sklearn: {accuracy:.6f}")
print(f"  manual:  {manual_accuracy:.6f} (TP+TN)/Total = ({tp}+{tn})/{tp+tn+fp+fn}")
if abs(accuracy - manual_accuracy) > 1e-10:
    print(f"  ERROR: Mismatch!")
else:
    print(f"  OK: Matches")

print(f"\nPrecision:")
print(f"  sklearn: {precision:.6f}")
print(f"  manual:  {manual_precision:.6f} TP/(TP+FP) = {tp}/({tp}+{fp})")
if abs(precision - manual_precision) > 1e-10:
    print(f"  ERROR: Mismatch!")
else:
    print(f"  OK: Matches")

print(f"\nRecall:")
print(f"  sklearn: {recall:.6f}")
print(f"  manual:  {manual_recall:.6f} TP/(TP+FN) = {tp}/({tp}+{fn})")
if abs(recall - manual_recall) > 1e-10:
    print(f"  ERROR: Mismatch!")
else:
    print(f"  OK: Matches")

print(f"\nF1-score:")
print(f"  sklearn: {f1:.6f}")
print(f"  manual:  {manual_f1:.6f} 2*(P*R)/(P+R)")
if abs(f1 - manual_f1) > 1e-6:
    print(f"  ERROR: Mismatch!")
else:
    print(f"  OK: Matches")

print(f"\nROC-AUC: {roc_auc:.6f}")
print(f"  (cannot verify manually, but should be in [0, 1])")
if roc_auc < 0 or roc_auc > 1:
    print(f"  ERROR: Out of valid range!")
else:
    print(f"  OK: In valid range")

print("\n3. VERIFYING PER-CLASS METRICS")
print("-" * 70)

# high-risk class metrics
high_risk_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
high_risk_recall = tp / (tp + fn) if (tp + fn) > 0 else 0

# low-risk class metrics
low_risk_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
low_risk_recall = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"High-risk class:")
print(f"  Precision: {high_risk_precision:.4f} (TP/(TP+FP) = {tp}/({tp}+{fp}))")
print(f"  Recall:    {high_risk_recall:.4f} (TP/(TP+FN) = {tp}/({tp}+{fn}))")
print(f"  Note: High-risk precision should equal overall precision: {precision:.4f}")
if abs(high_risk_precision - precision) > 1e-6:
    print(f"  ERROR: High-risk precision doesn't match overall precision!")
else:
    print(f"  OK: High-risk precision matches overall precision")

print(f"\nLow-risk class:")
print(f"  Precision: {low_risk_precision:.4f} (TN/(TN+FN) = {tn}/({tn}+{fn}))")
print(f"  Recall:    {low_risk_recall:.4f} (TN/(TN+FP) = {tn}/({tn}+{fp}))")

print("\n4. VERIFYING SUMMARY NUMBERS")
print("-" * 70)

# check summary claims
print(f"Summary claims:")
print(f"  '94.4% accuracy' - Actual: {accuracy:.4f} = {accuracy*100:.1f}%")
if abs(accuracy - 0.9444) > 1e-3:
    print(f"  ERROR: Accuracy doesn't match!")
else:
    print(f"  OK: Matches")

print(f"  '93.6% recall' - Actual: {recall:.4f} = {recall*100:.1f}%")
if abs(recall - 0.9355) > 1e-3:
    print(f"  ERROR: Recall doesn't match!")
else:
    print(f"  OK: Matches (note: 0.9355 = 93.55%, rounded to 93.6%)")

print(f"  '96.7% precision' - Actual: {precision:.4f} = {precision*100:.1f}%")
if abs(precision - 0.9667) > 1e-3:
    print(f"  ERROR: Precision doesn't match!")
else:
    print(f"  OK: Matches")

print(f"  'only 3 were misclassified' - Actual: {fp + fn} (FP={fp} + FN={fn})")
if (fp + fn) != 3:
    print(f"  ERROR: Misclassification count doesn't match!")
else:
    print(f"  OK: Matches")

print(f"  '2 false negatives' - Actual: {fn}")
if fn != 2:
    print(f"  ERROR: False negatives count doesn't match!")
else:
    print(f"  OK: Matches")

print(f"  '1 false positive' - Actual: {fp}")
if fp != 1:
    print(f"  ERROR: False positives count doesn't match!")
else:
    print(f"  OK: Matches")

print("\n5. VERIFYING ERROR ANALYSIS")
print("-" * 70)

# identify actual errors
test_df_with_pred = test_df.copy()
test_df_with_pred['predicted'] = y_pred
test_df_with_pred['predicted_proba'] = y_pred_proba

false_negatives = test_df_with_pred[
    (test_df_with_pred['corruption_risk'] == 1) & 
    (test_df_with_pred['predicted'] == 0)
]

false_positives = test_df_with_pred[
    (test_df_with_pred['corruption_risk'] == 0) & 
    (test_df_with_pred['predicted'] == 1)
]

print(f"False negatives count: {len(false_negatives)} (should be {fn})")
if len(false_negatives) != fn:
    print(f"  ERROR: Count mismatch!")
else:
    print(f"  OK: Matches")

print(f"False positives count: {len(false_positives)} (should be {fp})")
if len(false_positives) != fp:
    print(f"  ERROR: Count mismatch!")
else:
    print(f"  OK: Matches")

# verify we can get country/year info
full_df = pd.read_csv('data/processed/final_training_data.csv')
merged_test = test_df_with_pred.merge(
    full_df[feature_names + ['Country', 'Year', 'corruption_risk']],
    on=feature_names + ['corruption_risk'],
    how='left',
    suffixes=('', '_full')
)

if merged_test['Country'].isna().any():
    merged_test = test_df_with_pred.merge(
        full_df[feature_names + ['Country', 'Year']],
        on=feature_names,
        how='left'
    )

fn_with_info = merged_test[
    (merged_test['corruption_risk'] == 1) & 
    (merged_test['predicted'] == 0)
]

fp_with_info = merged_test[
    (merged_test['corruption_risk'] == 0) & 
    (merged_test['predicted'] == 1)
]

print(f"\nFalse negatives with country/year info: {len(fn_with_info)}")
if len(fn_with_info) > 0:
    print(f"  Countries: {', '.join(fn_with_info['Country'].dropna().unique())}")

print(f"False positives with country/year info: {len(fp_with_info)}")
if len(fp_with_info) > 0:
    print(f"  Countries: {', '.join(fp_with_info['Country'].dropna().unique())}")

print("\n6. VERIFYING FEATURE IMPORTANCE")
print("-" * 70)

feature_importance = dt_model.feature_importances_
used_features = [f for f, imp in zip(feature_names, feature_importance) if imp > 1e-10]

print(f"Features used: {len(used_features)}")
print(f"Expected: 3 (Poverty, Debt, Government Spending)")

if len(used_features) != 3:
    print(f"  ERROR: Wrong number of features!")
else:
    print(f"  OK: Matches")

expected_features = ['Poverty_Headcount_Ratio', 'External_Debt_perc_GNI', 'Govt_Expenditure_perc_GDP']
if set(used_features) != set(expected_features):
    print(f"  ERROR: Features don't match expected!")
    print(f"    Found: {used_features}")
    print(f"    Expected: {expected_features}")
else:
    print(f"  OK: Features match")

# verify importance values
feature_imp_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(f"\nFeature importance values:")
for _, row in feature_imp_df[feature_imp_df['importance'] > 1e-10].iterrows():
    print(f"  {row['feature']:30s}: {row['importance']:.4f}")

importance_sum = np.sum(feature_importance)
print(f"\nSum of importance: {importance_sum:.6f} (should be 1.0)")
if abs(importance_sum - 1.0) > 1e-10:
    print(f"  ERROR: Doesn't sum to 1.0!")
else:
    print(f"  OK: Sums to 1.0")

print("\n" + "="*70)
print("VERIFICATION COMPLETE")
print("="*70)

