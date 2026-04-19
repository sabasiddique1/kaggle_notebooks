#!/usr/bin/env python3
"""
OPTIMIZED Pipeline to Achieve 0.78 AUC
Based on best notebook (0.782 AUC) analysis
"""
import numpy as np
import pandas as pd
import warnings
import os
import sys
from pathlib import Path
import time

warnings.filterwarnings('ignore')

print("=" * 70)
print("🚀 OPTIMIZED PIPELINE FOR 0.78 AUC")
print("=" * 70)

os.chdir('/Users/saba/Desktop/diabetes-kaggle-comp')

# ============================================================================
# STEP 1: Load Data + External Dataset (CRITICAL!)
# ============================================================================
print("\n📊 STEP 1: Loading Data + External Dataset...")
print("-" * 70)

train = pd.read_csv('playground-series-s5e12/train.csv')
test = pd.read_csv('playground-series-s5e12/test.csv')

# Merge external dataset (CRITICAL for 0.78!)
try:
    orig_paths = [
        'diabetes-health-indicators-dataset/diabetes_binary_health_indicators_BRFSS2015.csv',
        'diabetes-health-indicators-dataset/diabetes_dataset.csv',
    ]
    orig = None
    for path in orig_paths:
        if os.path.exists(path):
            orig = pd.read_csv(path)
            break
    
    if orig is not None:
        # Ensure same columns
        orig['id'] = orig.index
        common_cols = [col for col in train.columns if col in orig.columns]
        if 'diagnosed_diabetes' in common_cols:
            orig = orig[common_cols]
            train = pd.concat([train, orig], axis=0).reset_index(drop=True)
            print(f"✅ External dataset merged! New train shape: {train.shape}")
        else:
            print("⚠️ External dataset columns don't match")
    else:
        print("⚠️ External dataset not found")
except Exception as e:
    print(f"⚠️ Could not merge external dataset: {e}")

print(f"✅ Train: {train.shape}")
print(f"✅ Test: {test.shape}")

# ============================================================================
# STEP 2: Feature Engineering
# ============================================================================
print("\n🔧 STEP 2: Feature Engineering...")
print("-" * 70)

def advanced_feature_engineering(df):
    df = df.copy()
    df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]).astype(int)
    df['chol_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1e-5)
    df['total_chol_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1e-5)
    df['lipid_ratio'] = df['triglycerides'] / (df['hdl_cholesterol'] + 1)
    df['bp_category'] = 0
    df.loc[(df['systolic_bp'] >= 130) | (df['diastolic_bp'] >= 80), 'bp_category'] = 1
    df.loc[(df['systolic_bp'] >= 140) | (df['diastolic_bp'] >= 90), 'bp_category'] = 2
    df['bp_ratio'] = df['systolic_bp'] / (df['diastolic_bp'] + 1)
    df['hypertension'] = ((df['systolic_bp'] >= 130) | (df['diastolic_bp'] >= 80)).astype(int)
    df['age_category'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3]).astype(int)
    df['medical_risk'] = (df['family_history_diabetes'] * 0.3 + 
                         df['hypertension_history'] * 0.3 + 
                         df['cardiovascular_history'] * 0.4)
    median_activity = df['physical_activity_minutes_per_week'].median()
    df['lifestyle_risk'] = (
        (df['smoking_status'] == 'Current').astype(int) * 0.4 + 
        (df['physical_activity_minutes_per_week'] < median_activity).astype(int) * 0.3 + 
        (df['bmi'] > 30).astype(int) * 0.3
    )
    df['age_bmi'] = df['age'] * df['bmi'] / 100
    df['age_chol'] = df['age'] * df['cholesterol_total'] / 100
    df['bmi_chol'] = df['bmi'] * df['cholesterol_total'] / 100
    df['family_age'] = df['family_history_diabetes'] * df['age'] / 10
    df['bp_bmi'] = df['systolic_bp'] * df['bmi'] / 100
    df['bmi_squared'] = df['bmi'] ** 2 / 100
    df['chol_squared'] = df['cholesterol_total'] ** 2 / 1000
    df['age_squared'] = df['age'] ** 2 / 1000
    return df

train_fe = advanced_feature_engineering(train)
test_fe = advanced_feature_engineering(test)
print(f"✅ Created {len(train_fe.columns) - len(train.columns)} new features")

# ============================================================================
# STEP 3: Target Encoding
# ============================================================================
print("\n🎯 STEP 3: Target Encoding...")
print("-" * 70)

from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin

class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_encode, cv=5, smooth='auto', drop_original=False):
        self.cols_to_encode = cols_to_encode
        self.cv = cv
        self.smooth = smooth
        self.drop_original = drop_original
        self.mappings_ = {}
        self.global_mean_ = None

    def fit(self, X, y):
        temp_df = X.copy()
        temp_df['target'] = y
        self.global_mean_ = y.mean()
        for col in self.cols_to_encode:
            mapping = temp_df.groupby(col)['target'].mean()
            self.mappings_[col] = mapping
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in self.cols_to_encode:
            new_col_name = f'TE_{col}'
            X_transformed[new_col_name] = X[col].map(self.mappings_[col])
            X_transformed[new_col_name].fillna(self.global_mean_, inplace=True)
        return X_transformed

    def fit_transform(self, X, y):
        self.fit(X, y)
        encoded_features = pd.DataFrame(index=X.index)
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            temp_df_train = X_train.copy()
            temp_df_train['target'] = y_train

            for col in self.cols_to_encode:
                new_col_name = f'TE_{col}'
                fold_global_mean = y_train.mean()
                mapping = temp_df_train.groupby(col)['target'].mean()
                
                if self.smooth == 'auto':
                    counts = temp_df_train.groupby(col)['target'].count()
                    variance_between = mapping.var()
                    avg_variance_within = temp_df_train.groupby(col)['target'].var().mean()
                    m = avg_variance_within / variance_between if variance_between > 0 else 0
                    smoothed_mapping = (counts * mapping + m * fold_global_mean) / (counts + m)
                    encoded_values = X_val[col].map(smoothed_mapping)
                else:
                    encoded_values = X_val[col].map(mapping)
                
                encoded_features.loc[X_val.index, new_col_name] = encoded_values.fillna(fold_global_mean)

        X_transformed = X.copy()
        for col in encoded_features.columns:
            X_transformed[col] = encoded_features[col]
        return X_transformed

# Prepare data - KEEP ID COLUMN!
y = train_fe['diagnosed_diabetes']
X = train_fe.drop(columns=['diagnosed_diabetes'])  # Keep 'id'!
X_test = test_fe  # Keep 'id'!

# Label encode categoricals
from sklearn.preprocessing import LabelEncoder
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([X[col], X_test[col]], axis=0).astype(str)
    le.fit(combined)
    X[col] = le.transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

# Identify integer columns for target encoding (including ID!)
int_cols = X.select_dtypes(include=['int64', 'int32']).columns.tolist()
int_cols = [col for col in int_cols if 'diagnosed' not in col.lower()]

print(f"✅ Prepared: X={X.shape}, y={y.shape}, X_test={X_test.shape}")
print(f"✅ Integer columns for TE: {len(int_cols)} (including ID)")

# ============================================================================
# STEP 4: OPTIMIZED Model Training
# ============================================================================
print("\n🤖 STEP 4: OPTIMIZED Model Training (5-Fold CV)...")
print("-" * 70)
print("🎯 Using OPTIMIZED parameters for 0.78 AUC:")
print("   - CatBoost: 12000 iterations (vs 2000)")
print("   - Early stopping: 300 rounds (vs 200)")
print("   - use_best_model: True")
print("   - Full dataset (not sample)")

USE_SAMPLE = False  # Use full dataset for 0.78!

if USE_SAMPLE:
    SAMPLE_SIZE = 100000
    print(f"📊 Using sample: {SAMPLE_SIZE} samples")
    X_sample = X.sample(n=min(SAMPLE_SIZE, len(X)), random_state=42)
    y_sample = y.loc[X_sample.index]
else:
    X_sample = X
    y_sample = y
    print(f"📊 Using FULL dataset: {len(X_sample):,} samples")

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_preds_cat = np.zeros(len(X_sample))
test_preds_cat = np.zeros(len(X_test))
scores_cat = []

start_time = time.time()

for fold, (train_idx, val_idx) in enumerate(skf.split(X_sample, y_sample), 1):
    print(f"\n📊 Fold {fold}/{n_folds}")
    
    X_train, X_val = X_sample.iloc[train_idx], X_sample.iloc[val_idx]
    y_train, y_val = y_sample.iloc[train_idx], y_sample.iloc[val_idx]
    
    # Apply Target Encoding
    TE = TargetEncoder(cols_to_encode=int_cols, cv=5, smooth='auto')
    X_train_te = TE.fit_transform(X_train, y_train)
    X_val_te = TE.transform(X_val)
    X_test_te = TE.transform(X_test)
    
    # OPTIMIZED CatBoost (for 0.78 AUC)
    try:
        from catboost import CatBoostClassifier
        cat_model = CatBoostClassifier(
            iterations=12000,  # OPTIMIZED: Increased from 2000
            learning_rate=0.01,
            depth=3,
            l2_leaf_reg=3,
            bagging_temperature=1,
            random_seed=42 + fold,
            eval_metric='AUC',
            use_best_model=True,  # OPTIMIZED: Add this
            verbose=False,
            early_stopping_rounds=300  # OPTIMIZED: Increased from 200
        )
        cat_model.fit(
            X_train_te, y_train,
            eval_set=(X_val_te, y_val),
            verbose=False
        )
        oof_preds_cat[val_idx] = cat_model.predict_proba(X_val_te)[:, 1]
        test_preds_cat += cat_model.predict_proba(X_test_te)[:, 1] / n_folds
        score_cat = roc_auc_score(y_val, oof_preds_cat[val_idx])
        scores_cat.append(score_cat)
        print(f"   CatBoost AUC: {score_cat:.5f}")
    except Exception as e:
        print(f"   CatBoost: Error - {e}")
        import traceback
        traceback.print_exc()

elapsed_time = time.time() - start_time

# ============================================================================
# STEP 5: Results
# ============================================================================
print("\n" + "=" * 70)
print("📊 RESULTS")
print("=" * 70)

cv_cat = roc_auc_score(y_sample, oof_preds_cat) if len(scores_cat) > 0 else 0
std_cat = np.std(scores_cat) if len(scores_cat) > 0 else 0.0

print(f"\nCross-Validation Scores ({'FULL' if not USE_SAMPLE else 'Sample'} Dataset):")
print(f"  CatBoost  CV AUC: {cv_cat:.5f} (std: {std_cat:.5f})")
print(f"\n⏱️  Training Time: {elapsed_time/60:.1f} minutes")

if cv_cat >= 0.78:
    print(f"\n🎉 SUCCESS! Achieved target: {cv_cat:.5f} >= 0.78")
else:
    print(f"\n📈 Current: {cv_cat:.5f}, Target: 0.78, Gap: {0.78 - cv_cat:.5f}")

# Create submission
print("\n💾 Creating Submission File...")
print("-" * 70)

submission = pd.DataFrame({
    'id': test['id'] if 'id' in test.columns else range(len(test)),
    'diagnosed_diabetes': test_preds_cat
})

submission_path = 'submission_optimized_0.78.csv'
submission.to_csv(submission_path, index=False)

print(f"✅ Submission saved: {submission_path}")
print(f"✅ Predictions range: [{test_preds_cat.min():.4f}, {test_preds_cat.max():.4f}]")
print(f"✅ Mean prediction: {test_preds_cat.mean():.4f}")

print("\n" + "=" * 70)
print("✅ OPTIMIZED PIPELINE COMPLETE!")
print("=" * 70)
print(f"\n📈 Expected Performance:")
print(f"   CV AUC: {cv_cat:.5f}")
print(f"   Target: 0.78")
print(f"   Status: {'✅ ACHIEVED!' if cv_cat >= 0.78 else '📈 Close!'}")





