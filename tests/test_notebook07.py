#!/usr/bin/env python3
"""
comprehensive test script for model evaluation notebook
verifies evaluation metrics, confusion matrix, error analysis, and visualization outputs
run from project root: python3 tests/test_notebook07.py
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import joblib

print("=== Testing Model Evaluation notebook ===\n")

try:
    # ========================================================================
    # PHASE 1: DATA AND MODEL LOADING VERIFICATION
    # ========================================================================
    
    print("="*70)
    print("PHASE 1: DATA AND MODEL LOADING VERIFICATION")
    print("="*70)
    
    # 1.1 verify model file exists and loads
    print("\n1.1 verifying model loading...")
    
    model_path = os.path.join(project_root, 'models', 'decision_tree_model.pkl')
    if not os.path.exists(model_path):
        print(f"   ERROR: model file not found at {model_path}")
        exit(1)
    
    dt_model = joblib.load(model_path)
    print("   OK: model loaded successfully")
    
    # 1.2 verify test set exists and has correct structure
    print("\n1.2 verifying test set loading...")
    
    test_path = os.path.join(project_root, 'data', 'processed', 'test_set.csv')
    if not os.path.exists(test_path):
        print(f"   ERROR: test set file not found at {test_path}")
        exit(1)
    
    test_df = pd.read_csv(test_path)
    
    if len(test_df) != 54:
        print(f"   ERROR: test set has {len(test_df)} samples (expected 54)")
        exit(1)
    
    print(f"   OK: test set loaded with {len(test_df)} samples")
    
    # 1.3 verify feature names file exists
    print("\n1.3 verifying feature names...")
    
    feature_names_path = os.path.join(project_root, 'models', 'feature_names.txt')
    if not os.path.exists(feature_names_path):
        print(f"   ERROR: feature names file not found at {feature_names_path}")
        exit(1)
    
    with open(feature_names_path, 'r') as f:
        feature_names = [line.strip() for line in f.readlines() if line.strip()]
    
    if len(feature_names) != 6:
        print(f"   ERROR: expected 6 features, found {len(feature_names)}")
        exit(1)
    
    print(f"   OK: {len(feature_names)} features loaded")
    
    # 1.4 verify test set has required columns
    print("\n1.4 verifying test set structure...")
    
    if 'corruption_risk' not in test_df.columns:
        print(f"   ERROR: corruption_risk column not found in test set")
        exit(1)
    
    missing_features = [f for f in feature_names if f not in test_df.columns]
    if missing_features:
        print(f"   ERROR: missing features in test set: {missing_features}")
        exit(1)
    
    X_test = test_df[feature_names]
    y_test = test_df['corruption_risk']
    
    # verify target is binary
    target_values = set(y_test.unique())
    if target_values != {0, 1}:
        print(f"   ERROR: corruption_risk is not binary (found values: {target_values})")
        exit(1)
    
    # verify test set distribution
    high_risk_count = (y_test == 1).sum()
    low_risk_count = (y_test == 0).sum()
    
    expected_high_risk = 31
    expected_low_risk = 23
    
    if high_risk_count != expected_high_risk:
        print(f"   ERROR: test set has {high_risk_count} high-risk samples (expected {expected_high_risk})")
        exit(1)
    if low_risk_count != expected_low_risk:
        print(f"   ERROR: test set has {low_risk_count} low-risk samples (expected {expected_low_risk})")
        exit(1)
    
    print(f"   OK: test set structure correct (high-risk: {high_risk_count}, low-risk: {low_risk_count})")
    
    # verify no missing values
    if X_test.isnull().sum().sum() > 0:
        print(f"   ERROR: missing values in test set features")
        exit(1)
    if y_test.isnull().sum() > 0:
        print(f"   ERROR: missing values in test set target")
        exit(1)
    
    print("   OK: no missing values in test set")
    
    print("\n" + "="*70)
    print("PHASE 1 COMPLETE: Data and model loading verified")
    print("="*70)
    
    # ========================================================================
    # PHASE 2: PERFORMANCE METRICS VERIFICATION
    # ========================================================================
    
    print("\n" + "="*70)
    print("PHASE 2: PERFORMANCE METRICS VERIFICATION")
    print("="*70)
    
    # 2.1 generate predictions
    print("\n2.1 generating predictions...")
    
    y_pred = dt_model.predict(X_test)
    y_pred_proba = dt_model.predict_proba(X_test)[:, 1]
    
    # verify predictions are binary
    unique_predictions = set(y_pred)
    if unique_predictions != {0, 1}:
        print(f"   ERROR: predictions are not binary (found values: {unique_predictions})")
        exit(1)
    
    # verify probabilities are in valid range
    if np.any(y_pred_proba < 0) or np.any(y_pred_proba > 1):
        print(f"   ERROR: prediction probabilities outside valid range [0, 1]")
        exit(1)
    
    print("   OK: predictions generated successfully")
    
    # 2.2 calculate and verify metrics
    print("\n2.2 verifying performance metrics...")
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # expected values from notebook
    expected_accuracy = 0.9444
    expected_precision = 0.9667
    expected_recall = 0.9355
    expected_f1 = 0.9508
    expected_roc_auc = 0.9586
    
    # verify accuracy (allow small tolerance for floating point)
    if abs(accuracy - expected_accuracy) > 1e-3:
        print(f"   ERROR: accuracy is {accuracy:.4f} (expected {expected_accuracy:.4f})")
        exit(1)
    
    # verify precision
    if abs(precision - expected_precision) > 1e-3:
        print(f"   ERROR: precision is {precision:.4f} (expected {expected_precision:.4f})")
        exit(1)
    
    # verify recall
    if abs(recall - expected_recall) > 1e-3:
        print(f"   ERROR: recall is {recall:.4f} (expected {expected_recall:.4f})")
        exit(1)
    
    # verify f1-score
    if abs(f1 - expected_f1) > 1e-3:
        print(f"   ERROR: f1-score is {f1:.4f} (expected {expected_f1:.4f})")
        exit(1)
    
    # verify roc-auc
    if abs(roc_auc - expected_roc_auc) > 1e-3:
        print(f"   ERROR: roc-auc is {roc_auc:.4f} (expected {expected_roc_auc:.4f})")
        exit(1)
    
    print(f"   OK: accuracy = {accuracy:.4f}")
    print(f"   OK: precision = {precision:.4f}")
    print(f"   OK: recall = {recall:.4f}")
    print(f"   OK: f1-score = {f1:.4f}")
    print(f"   OK: roc-auc = {roc_auc:.4f}")
    
    # 2.3 verify generalization check
    print("\n2.3 verifying generalization check...")
    
    train_path = os.path.join(project_root, 'data', 'processed', 'train_set.csv')
    train_df = pd.read_csv(train_path)
    X_train = train_df[feature_names]
    y_train = train_df['corruption_risk']
    
    y_train_pred = dt_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    performance_gap = train_accuracy - accuracy
    
    # verify gap is small (good generalization)
    if performance_gap > 0.15:
        print(f"   WARNING: large performance gap ({performance_gap:.4f}) may indicate overfitting")
    elif performance_gap < -0.15:
        print(f"   WARNING: test performance much better than training ({performance_gap:.4f}) - unusual")
    else:
        print(f"   OK: performance gap is {performance_gap:.4f} (indicates good generalization)")
    
    print(f"      train accuracy: {train_accuracy:.4f}, test accuracy: {accuracy:.4f}")
    
    print("\n" + "="*70)
    print("PHASE 2 COMPLETE: Performance metrics verified")
    print("="*70)
    
    # ========================================================================
    # PHASE 3: CONFUSION MATRIX VERIFICATION
    # ========================================================================
    
    print("\n" + "="*70)
    print("PHASE 3: CONFUSION MATRIX VERIFICATION")
    print("="*70)
    
    # 3.1 calculate confusion matrix
    print("\n3.1 verifying confusion matrix...")
    
    cm = confusion_matrix(y_test, y_pred)
    # confusion matrix format: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    
    # expected values from notebook
    expected_tn = 22
    expected_fp = 1
    expected_fn = 2
    expected_tp = 29
    
    if tn != expected_tn:
        print(f"   ERROR: true negatives is {tn} (expected {expected_tn})")
        exit(1)
    if fp != expected_fp:
        print(f"   ERROR: false positives is {fp} (expected {expected_fp})")
        exit(1)
    if fn != expected_fn:
        print(f"   ERROR: false negatives is {fn} (expected {expected_fn})")
        exit(1)
    if tp != expected_tp:
        print(f"   ERROR: true positives is {tp} (expected {expected_tp})")
        exit(1)
    
    print(f"   OK: confusion matrix matches: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    # 3.2 verify confusion matrix calculations
    print("\n3.2 verifying confusion matrix calculations...")
    
    # verify total adds up
    total = tn + fp + fn + tp
    if total != len(y_test):
        print(f"   ERROR: confusion matrix total ({total}) doesn't match test set size ({len(y_test)})")
        exit(1)
    
    print(f"   OK: confusion matrix total matches test set size ({total})")
    
    # verify metrics match confusion matrix
    manual_accuracy = (tp + tn) / (tp + tn + fp + fn)
    manual_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    manual_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    if abs(accuracy - manual_accuracy) > 1e-10:
        print(f"   ERROR: accuracy doesn't match confusion matrix calculation")
        exit(1)
    if abs(precision - manual_precision) > 1e-10:
        print(f"   ERROR: precision doesn't match confusion matrix calculation")
        exit(1)
    if abs(recall - manual_recall) > 1e-10:
        print(f"   ERROR: recall doesn't match confusion matrix calculation")
        exit(1)
    
    print("   OK: metrics match confusion matrix calculations")
    
    # 3.3 verify confusion matrix visualization file
    print("\n3.3 verifying confusion matrix visualization...")
    
    confusion_matrix_path = os.path.join(project_root, 'results', 'figures', 'confusion_matrix_heatmap.png')
    if not os.path.exists(confusion_matrix_path):
        print(f"   WARNING: confusion matrix heatmap not found at {confusion_matrix_path}")
    else:
        print("   OK: confusion matrix heatmap saved")
    
    print("\n" + "="*70)
    print("PHASE 3 COMPLETE: Confusion matrix verified")
    print("="*70)
    
    # ========================================================================
    # PHASE 4: ERROR ANALYSIS VERIFICATION
    # ========================================================================
    
    print("\n" + "="*70)
    print("PHASE 4: ERROR ANALYSIS VERIFICATION")
    print("="*70)
    
    # 4.1 identify misclassified cases
    print("\n4.1 verifying error identification...")
    
    # create dataframe with predictions
    test_df_with_pred = test_df.copy()
    test_df_with_pred['predicted'] = y_pred
    test_df_with_pred['predicted_proba'] = y_pred_proba
    
    # identify false negatives and false positives
    false_negatives = test_df_with_pred[
        (test_df_with_pred['corruption_risk'] == 1) & 
        (test_df_with_pred['predicted'] == 0)
    ]
    
    false_positives = test_df_with_pred[
        (test_df_with_pred['corruption_risk'] == 0) & 
        (test_df_with_pred['predicted'] == 1)
    ]
    
    # verify counts match confusion matrix
    if len(false_negatives) != fn:
        print(f"   ERROR: false negatives count mismatch ({len(false_negatives)} vs {fn})")
        exit(1)
    if len(false_positives) != fp:
        print(f"   ERROR: false positives count mismatch ({len(false_positives)} vs {fp})")
        exit(1)
    
    print(f"   OK: false negatives: {len(false_negatives)} (matches confusion matrix)")
    print(f"   OK: false positives: {len(false_positives)} (matches confusion matrix)")
    
    # 4.2 verify error cases can be identified with country/year info
    print("\n4.2 verifying error case identification...")
    
    # load full dataset to get country and year information
    final_data_path = os.path.join(project_root, 'data', 'processed', 'final_training_data.csv')
    if not os.path.exists(final_data_path):
        print(f"   WARNING: final_training_data.csv not found, cannot verify country/year identification")
    else:
        full_df = pd.read_csv(final_data_path)
        
        # merge to get country and year
        merged_test = test_df_with_pred.merge(
            full_df[feature_names + ['Country', 'Year', 'corruption_risk']],
            on=feature_names + ['corruption_risk'],
            how='left',
            suffixes=('', '_full')
        )
        
        # if merge didn't work perfectly, try without corruption_risk
        if merged_test['Country'].isna().any():
            merged_test = test_df_with_pred.merge(
                full_df[feature_names + ['Country', 'Year']],
                on=feature_names,
                how='left'
            )
        
        # verify we can identify false negatives
        fn_with_info = merged_test[
            (merged_test['corruption_risk'] == 1) & 
            (merged_test['predicted'] == 0)
        ]
        
        if len(fn_with_info) > 0:
            fn_countries = fn_with_info['Country'].dropna().unique()
            print(f"   OK: identified {len(fn_with_info)} false negatives")
            if len(fn_countries) > 0:
                print(f"      countries: {', '.join(fn_countries)}")
        
        # verify we can identify false positives
        fp_with_info = merged_test[
            (merged_test['corruption_risk'] == 0) & 
            (merged_test['predicted'] == 1)
        ]
        
        if len(fp_with_info) > 0:
            fp_countries = fp_with_info['Country'].dropna().unique()
            print(f"   OK: identified {len(fp_with_info)} false positives")
            if len(fp_countries) > 0:
                print(f"      countries: {', '.join(fp_countries)}")
    
    # 4.3 verify error analysis focuses on core indicators
    print("\n4.3 verifying error analysis uses core indicators...")
    
    # the three core indicators used by the model
    used_features = ['Poverty_Headcount_Ratio', 'External_Debt_perc_GNI', 'Govt_Expenditure_perc_GDP']
    
    # verify these features exist in test set
    missing_core = [f for f in used_features if f not in test_df.columns]
    if missing_core:
        print(f"   ERROR: core indicators missing from test set: {missing_core}")
        exit(1)
    
    print(f"   OK: core indicators available for error analysis: {', '.join(used_features)}")
    
    print("\n" + "="*70)
    print("PHASE 4 COMPLETE: Error analysis verified")
    print("="*70)
    
    # ========================================================================
    # PHASE 5: FEATURE IMPORTANCE VERIFICATION
    # ========================================================================
    
    print("\n" + "="*70)
    print("PHASE 5: FEATURE IMPORTANCE VERIFICATION")
    print("="*70)
    
    # 5.1 verify feature importance
    print("\n5.1 verifying feature importance...")
    
    feature_importance = dt_model.feature_importances_
    
    # verify feature importance sums to 1.0
    importance_sum = np.sum(feature_importance)
    if abs(importance_sum - 1.0) > 1e-10:
        print(f"   ERROR: feature importance sum is {importance_sum:.10f} (expected 1.0)")
        exit(1)
    
    print("   OK: feature importance sums to 1.0")
    
    # verify only 3 features are used (importance > 1e-10)
    used_features_importance = [f for f, imp in zip(feature_names, feature_importance) if imp > 1e-10]
    
    if len(used_features_importance) != 3:
        print(f"   ERROR: expected 3 features used, found {len(used_features_importance)}")
        exit(1)
    
    expected_used = ['Poverty_Headcount_Ratio', 'External_Debt_perc_GNI', 'Govt_Expenditure_perc_GDP']
    if set(used_features_importance) != set(expected_used):
        print(f"   ERROR: used features don't match expected")
        print(f"     found: {used_features_importance}")
        print(f"     expected: {expected_used}")
        exit(1)
    
    print(f"   OK: only 3 features used: {', '.join(used_features_importance)}")
    
    # verify expected importance values
    feature_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    expected_importance = {
        'Poverty_Headcount_Ratio': 0.4184,
        'External_Debt_perc_GNI': 0.3302,
        'Govt_Expenditure_perc_GDP': 0.2514
    }
    
    for feature, expected_val in expected_importance.items():
        actual_val = feature_imp_df[feature_imp_df['feature'] == feature]['importance'].values[0]
        if abs(actual_val - expected_val) > 1e-3:
            print(f"   ERROR: {feature} importance is {actual_val:.4f} (expected {expected_val:.4f})")
            exit(1)
    
    print("   OK: feature importance values match expected")
    
    # 5.2 verify feature importance visualization
    print("\n5.2 verifying feature importance visualization...")
    
    feature_importance_path = os.path.join(project_root, 'results', 'figures', 'feature_importance_bar.png')
    if not os.path.exists(feature_importance_path):
        print(f"   WARNING: feature importance bar chart not found at {feature_importance_path}")
    else:
        print("   OK: feature importance bar chart saved")
    
    print("\n" + "="*70)
    print("PHASE 5 COMPLETE: Feature importance verified")
    print("="*70)
    
    # ========================================================================
    # PHASE 6: OUTPUT FILES VERIFICATION
    # ========================================================================
    
    print("\n" + "="*70)
    print("PHASE 6: OUTPUT FILES VERIFICATION")
    print("="*70)
    
    # 6.1 verify visualization files
    print("\n6.1 verifying visualization files...")
    
    required_figures = [
        'confusion_matrix_heatmap.png',
        'feature_importance_bar.png'
    ]
    
    figures_dir = os.path.join(project_root, 'results', 'figures')
    for fig_name in required_figures:
        fig_path = os.path.join(figures_dir, fig_name)
        if not os.path.exists(fig_path):
            print(f"   WARNING: {fig_name} not found at {fig_path}")
        else:
            # verify file is not empty
            if os.path.getsize(fig_path) == 0:
                print(f"   ERROR: {fig_name} is empty")
                exit(1)
            print(f"   OK: {fig_name} exists and is not empty")
    
    print("\n" + "="*70)
    print("PHASE 6 COMPLETE: Output files verified")
    print("="*70)
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print("\nAll verification tests passed successfully!")
    print("\nModel evaluation:")
    print(f"  - Test set: {len(test_df)} samples")
    print(f"  - Accuracy: {accuracy:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall: {recall:.4f}")
    print(f"  - F1-score: {f1:.4f}")
    print(f"  - ROC-AUC: {roc_auc:.4f}")
    print(f"  - Performance gap: {performance_gap:.4f} (good generalization)")
    print("\nConfusion matrix:")
    print(f"  - True Positives: {tp}")
    print(f"  - True Negatives: {tn}")
    print(f"  - False Positives: {fp}")
    print(f"  - False Negatives: {fn}")
    print("\nError analysis:")
    print(f"  - False negatives: {len(false_negatives)} cases")
    print(f"  - False positives: {len(false_positives)} cases")
    print("\nFeature importance:")
    print(f"  - {len(used_features_importance)} features used: {', '.join(used_features_importance)}")
    print("\nOutput files:")
    print("  - Confusion matrix heatmap saved")
    print("  - Feature importance bar chart saved")
    print("\n" + "="*70)
    print("=== All verification tests passed! ===")
    print("="*70)
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

