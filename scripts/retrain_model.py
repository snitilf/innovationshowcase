#!/usr/bin/env python3
"""
retrain decision tree model with only predictive features (economic + sentiment).
verifies that the model uses multiple features, not just one.
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib
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

# load training and test sets
train_df = pd.read_csv('data/processed/train_set.csv')
test_df = pd.read_csv('data/processed/test_set.csv')

# load feature names
with open('models/feature_names.txt', 'r') as f:
    feature_names = [line.strip() for line in f.readlines() if line.strip()]

print("="*70)
print("MODEL RETRAINING")
print("="*70)
print(f"\ntraining set shape: {train_df.shape}")
print(f"test set shape: {test_df.shape}")
print(f"\npredictive features ({len(feature_names)}):")
for i, feature in enumerate(feature_names, 1):
    print(f"  {i}. {feature}")

# extract feature matrix and target
X_train = train_df[feature_names]
y_train = train_df['corruption_risk']
X_test = test_df[feature_names]
y_test = test_df['corruption_risk']

print(f"\nfeature matrix shapes:")
print(f"  X_train: {X_train.shape}")
print(f"  X_test: {X_test.shape}")

# initialize and train decision tree
print(f"\ntraining decision tree model...")
dt_model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42
)

dt_model.fit(X_train, y_train)
print("✓ model training complete")

# generate predictions
y_pred = dt_model.predict(X_test)
y_pred_proba = dt_model.predict_proba(X_test)[:, 1]

# calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "="*70)
print("MODEL PERFORMANCE - TEST SET")
print("="*70)
print(f"\naccuracy:  {accuracy:.4f}")
print(f"precision: {precision:.4f}")
print(f"recall:    {recall:.4f}")
print(f"f1-score:  {f1:.4f}")
print(f"roc-auc:   {roc_auc:.4f}")

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nconfusion matrix:")
print(f"                predicted")
print(f"              low  high")
print(f"actual low   {cm[0,0]:4d}  {cm[0,1]:4d}")
print(f"       high  {cm[1,0]:4d}  {cm[1,1]:4d}")

# feature importance analysis
print("\n" + "="*70)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*70)

feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': dt_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nfeature importance (sorted):")
for i, row in feature_importance.iterrows():
    print(f"  {row['feature']:30s} {row['importance']:.6f}")

# count features actually used (importance > 1e-10)
used_features = feature_importance[feature_importance['importance'] > 1e-10]
print(f"\nfeatures actually used by model: {len(used_features)} out of {len(feature_names)}")
if len(used_features) < len(feature_names):
    unused = feature_importance[feature_importance['importance'] <= 1e-10]
    print(f"unused features:")
    for _, row in unused.iterrows():
        print(f"  - {row['feature']}")

# verify model uses multiple features (not just one)
if len(used_features) > 1:
    print(f"\n✓ model uses multiple features (good - no circular reasoning)")
    print(f"  top 3 features:")
    for i, (_, row) in enumerate(used_features.head(3).iterrows(), 1):
        print(f"    {i}. {row['feature']:30s} ({row['importance']:.4f})")
elif len(used_features) == 1:
    print(f"\n⚠️  warning: model uses only 1 feature")
    print(f"  this may indicate the feature is too predictive (possible data leakage)")
    print(f"  feature: {used_features.iloc[0]['feature']}")
else:
    print(f"\n⚠️  error: model uses no features")

# save model
os.makedirs('models', exist_ok=True)
joblib.dump(dt_model, 'models/decision_tree_model.pkl')
print(f"\n✓ saved model to models/decision_tree_model.pkl")

# save metrics
os.makedirs('results/tables', exist_ok=True)
metrics_df = pd.DataFrame({
    'metric': ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'],
    'value': [accuracy, precision, recall, f1, roc_auc]
})
metrics_df.to_csv('results/tables/model_performance.csv', index=False)
print(f"✓ saved metrics to results/tables/model_performance.csv")

# save feature importance
feature_importance.to_csv('results/tables/feature_importance.csv', index=False)
print(f"✓ saved feature importance to results/tables/feature_importance.csv")

print("\n" + "="*70)
print("✓ MODEL RETRAINING COMPLETE")
print("="*70)

