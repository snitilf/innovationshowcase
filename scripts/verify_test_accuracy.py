#!/usr/bin/env python3
"""
verification script to double-check test accuracy
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
venv_python = os.path.join(project_root, 'venv', 'bin', 'python3')
if os.path.exists(venv_python) and not sys.executable.startswith(os.path.join(project_root, 'venv')):
    os.execv(venv_python, [venv_python] + sys.argv)

import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

print("="*70)
print("VERIFICATION: Checking Test Accuracy")
print("="*70)

# load model and test data
model = joblib.load('models/decision_tree_model.pkl')
test_df = pd.read_csv('data/processed/test_set.csv')

with open('models/feature_names.txt', 'r') as f:
    features = [line.strip() for line in f.readlines()]

X_test = test_df[features]
y_test = test_df['corruption_risk']

# generate predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]

print("\n1. Confusion Matrix Verification:")
print(f"   Confusion Matrix:\n   {cm}")
print(f"   TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"   Expected: TN=22, FP=1, FN=2, TP=29")
if tn == 22 and fp == 1 and fn == 2 and tp == 29:
    print("   ✓ CORRECT")
else:
    print("   ✗ MISMATCH")

# calculate metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_pred_proba)

print("\n2. Metric Calculations:")
print(f"   Accuracy:  {acc:.10f}")
print(f"   Precision: {prec:.10f}")
print(f"   Recall:    {rec:.10f}")
print(f"   F1-score:  {f1:.10f}")
print(f"   ROC-AUC:   {roc:.10f}")

# manual calculations
manual_acc = (tp + tn) / (tp + tn + fp + fn)
manual_prec = tp / (tp + fp) if (tp + fp) > 0 else 0
manual_rec = tp / (tp + fn) if (tp + fn) > 0 else 0
manual_f1 = 2 * (manual_prec * manual_rec) / (manual_prec + manual_rec) if (manual_prec + manual_rec) > 0 else 0

print("\n3. Manual Calculations (verification):")
print(f"   Accuracy:  {manual_acc:.10f} = ({tp}+{tn})/{tp+tn+fp+fn} = {tp+tn}/{len(y_test)}")
print(f"   Precision: {manual_prec:.10f} = {tp}/({tp}+{fp}) = {tp}/{tp+fp}")
print(f"   Recall:    {manual_rec:.10f} = {tp}/({tp}+{fn}) = {tp}/{tp+fn}")
print(f"   F1-score:  {manual_f1:.10f} = 2*({manual_prec:.6f}*{manual_rec:.6f})/({manual_prec:.6f}+{manual_rec:.6f})")

# check if manual matches sklearn
print("\n4. Manual vs Sklearn Comparison:")
acc_match = abs(acc - manual_acc) < 1e-10
prec_match = abs(prec - manual_prec) < 1e-10
rec_match = abs(rec - manual_rec) < 1e-10
f1_match = abs(f1 - manual_f1) < 1e-6  # f1 can have slightly more tolerance

print(f"   Accuracy match:  {acc_match} (diff: {abs(acc - manual_acc):.2e})")
print(f"   Precision match: {prec_match} (diff: {abs(prec - manual_prec):.2e})")
print(f"   Recall match:    {rec_match} (diff: {abs(rec - manual_rec):.2e})")
print(f"   F1-score match:  {f1_match} (diff: {abs(f1 - manual_f1):.2e})")

if acc_match and prec_match and rec_match and f1_match:
    print("   ✓ All manual calculations match sklearn")
else:
    print("   ✗ Some calculations don't match")

# check saved metrics
saved = pd.read_csv('results/tables/model_performance.csv')
print("\n5. Saved Metrics Comparison:")
for _, row in saved.iterrows():
    metric = row['metric']
    saved_val = row['value']
    if metric == 'accuracy':
        actual_val = acc
    elif metric == 'precision':
        actual_val = prec
    elif metric == 'recall':
        actual_val = rec
    elif metric == 'f1_score':
        actual_val = f1
    elif metric == 'roc_auc':
        actual_val = roc
    else:
        continue
    
    diff = abs(actual_val - saved_val)
    match = diff < 1e-10
    status = "✓" if match else "✗"
    print(f"   {status} {metric:12s} actual={actual_val:.10f} saved={saved_val:.10f} diff={diff:.2e}")

# feature importance
importance = model.feature_importances_
imp_df = pd.DataFrame({'feature': features, 'importance': importance}).sort_values('importance', ascending=False)

print("\n6. Feature Importance Verification:")
print(f"   Sum: {np.sum(importance):.10f} (should be 1.0)")
if abs(np.sum(importance) - 1.0) < 1e-10:
    print("   ✓ Sum is 1.0")
else:
    print("   ✗ Sum is not 1.0")

saved_imp = pd.read_csv('results/tables/feature_importance.csv')
print("\n7. Feature Importance Values:")
for feature in features:
    actual = imp_df[imp_df['feature'] == feature]['importance'].values[0]
    saved_val = saved_imp[saved_imp['feature'] == feature]['importance'].values[0]
    diff = abs(actual - saved_val)
    match = diff < 1e-10
    status = "✓" if match else "✗"
    print(f"   {status} {feature:30s} actual={actual:.10f} saved={saved_val:.10f} diff={diff:.2e}")

# expected values from notebook
expected_importance = {
    'Poverty_Headcount_Ratio': 0.418435,
    'External_Debt_perc_GNI': 0.330193,
    'Govt_Expenditure_perc_GDP': 0.251372,
    'GDP_Growth_annual_perc': 0.0,
    'FDI_Inflows_perc_GDP': 0.0,
    'sentiment_score': 0.0
}

print("\n8. Feature Importance vs Expected (from notebook):")
all_match = True
for feature, expected_val in expected_importance.items():
    actual = imp_df[imp_df['feature'] == feature]['importance'].values[0]
    if expected_val == 0.0:
        match = abs(actual) < 1e-10
    else:
        match = abs(actual - expected_val) < 1e-6
    status = "✓" if match else "✗"
    print(f"   {status} {feature:30s} actual={actual:.10f} expected={expected_val:.6f}")
    if not match:
        all_match = False

# train/test split
train_df = pd.read_csv('data/processed/train_set.csv')
print("\n9. Train/Test Split Verification:")
print(f"   Train: {len(train_df)} rows (expected 212)")
print(f"   Test:  {len(test_df)} rows (expected 54)")
print(f"   Total: {len(train_df) + len(test_df)} rows (expected 266)")

train_match = len(train_df) == 212
test_match = len(test_df) == 54
total_match = (len(train_df) + len(test_df)) == 266

if train_match and test_match and total_match:
    print("   ✓ All split sizes match expected values")
else:
    print("   ✗ Some split sizes don't match")

# class distribution
train_y = train_df['corruption_risk']
test_y = test_df['corruption_risk']

print("\n10. Class Distribution:")
print(f"   Train: {train_y.sum()} high-risk, {len(train_y) - train_y.sum()} low-risk (expected: 123, 89)")
print(f"   Test:  {test_y.sum()} high-risk, {len(test_y) - test_y.sum()} low-risk (expected: 31, 23)")

train_high_match = train_y.sum() == 123
train_low_match = (len(train_y) - train_y.sum()) == 89
test_high_match = test_y.sum() == 31
test_low_match = (len(test_y) - test_y.sum()) == 23

if train_high_match and train_low_match and test_high_match and test_low_match:
    print("   ✓ All class distributions match expected values")
else:
    print("   ✗ Some class distributions don't match")

# model parameters
print("\n11. Model Parameters:")
print(f"   max_depth: {model.max_depth} (expected: 5)")
print(f"   min_samples_split: {model.min_samples_split} (expected: 10)")
print(f"   min_samples_leaf: {model.min_samples_leaf} (expected: 5)")
print(f"   class_weight: {model.class_weight} (expected: balanced)")
print(f"   random_state: {model.random_state} (expected: 42)")
print(f"   criterion: {model.criterion} (expected: gini)")
print(f"   splitter: {model.splitter} (expected: best)")

param_match = (
    model.max_depth == 5 and
    model.min_samples_split == 10 and
    model.min_samples_leaf == 5 and
    model.class_weight == 'balanced' and
    model.random_state == 42 and
    model.criterion == 'gini' and
    model.splitter == 'best'
)

if param_match:
    print("   ✓ All parameters match expected values")
else:
    print("   ✗ Some parameters don't match")

print("\n" + "="*70)
print("VERIFICATION SUMMARY")
print("="*70)

all_checks = [
    (tn == 22 and fp == 1 and fn == 2 and tp == 29, "Confusion matrix"),
    (acc_match and prec_match and rec_match and f1_match, "Metric calculations"),
    (abs(np.sum(importance) - 1.0) < 1e-10, "Feature importance sum"),
    (all_match, "Feature importance values"),
    (train_match and test_match and total_match, "Train/test split sizes"),
    (train_high_match and train_low_match and test_high_match and test_low_match, "Class distribution"),
    (param_match, "Model parameters")
]

all_passed = all(check[0] for check in all_checks)

for passed, check_name in all_checks:
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"   {status}: {check_name}")

print("\n" + "="*70)
if all_passed:
    print("✓ ALL VERIFICATIONS PASSED - Tests are accurate!")
else:
    print("✗ SOME VERIFICATIONS FAILED - Tests need review")
print("="*70)

