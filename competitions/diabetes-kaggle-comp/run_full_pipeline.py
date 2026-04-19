#!/usr/bin/env python3
"""
Run the comprehensive diabetes prediction pipeline
This script executes the full notebook pipeline and generates predictions
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
print("🚀 COMPREHENSIVE DIABETES PREDICTION PIPELINE")
print("=" * 70)

# Change to project directory
os.chdir('/Users/saba/Desktop/diabetes-kaggle-comp')

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("\n📊 STEP 1: Loading Data...")
print("-" * 70)

data_paths = [
    './playground-series-s5e12/train.csv',
    '../playground-series-s5e12/train.csv',
    '../../playground-series-s5e12/train.csv',
]

train_path = None
test_path = None

for path in data_paths:
    if os.path.exists(path):
        train_path = path
        test_path = path.replace('train.csv', 'test.csv')
        break

if not train_path or not os.path.exists(test_path):
    print("❌ Data files not found!")
    sys.exit(1)

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print(f"✅ Train: {train.shape}")
print(f"✅ Test: {test.shape}")
print(f"✅ Target distribution: {train['diagnosed_diabetes'].value_counts().to_dict()}")

# ============================================================================
# STEP 2: Feature Engineering
# ============================================================================
print("\n🔧 STEP 2: Feature Engineering...")
print("-" * 70)

def advanced_feature_engineering(df):
    """Create advanced features based on medical domain knowledge"""
    df = df.copy()
    
    # BMI Categories
    df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]).astype(int)
    
    # Cholesterol Ratios
    df['chol_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1e-5)
    df['total_chol_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1e-5)
    df['lipid_ratio'] = df['triglycerides'] / (df['hdl_cholesterol'] + 1)
    
    # Blood Pressure Categories
    df['bp_category'] = 0
    df.loc[(df['systolic_bp'] >= 130) | (df['diastolic_bp'] >= 80), 'bp_category'] = 1
    df.loc[(df['systolic_bp'] >= 140) | (df['diastolic_bp'] >= 90), 'bp_category'] = 2
    df['bp_ratio'] = df['systolic_bp'] / (df['diastolic_bp'] + 1)
    df['hypertension'] = ((df['systolic_bp'] >= 130) | (df['diastolic_bp'] >= 80)).astype(int)
    
    # Age Groups
    df['age_category'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3]).astype(int)
    
    # Medical Risk Score
    df['medical_risk'] = (df['family_history_diabetes'] * 0.3 + 
                         df['hypertension_history'] * 0.3 + 
                         df['cardiovascular_history'] * 0.4)
    
    # Lifestyle Risk Score
    median_activity = df['physical_activity_minutes_per_week'].median()
    df['lifestyle_risk'] = (
        (df['smoking_status'] == 'Current').astype(int) * 0.4 + 
        (df['physical_activity_minutes_per_week'] < median_activity).astype(int) * 0.3 + 
        (df['bmi'] > 30).astype(int) * 0.3
    )
    
    # Interaction Features
    df['age_bmi'] = df['age'] * df['bmi'] / 100
    df['age_chol'] = df['age'] * df['cholesterol_total'] / 100
    df['bmi_chol'] = df['bmi'] * df['cholesterol_total'] / 100
    df['family_age'] = df['family_history_diabetes'] * df['age'] / 10
    df['bp_bmi'] = df['systolic_bp'] * df['bmi'] / 100
    
    # Polynomial Features
    df['bmi_squared'] = df['bmi'] ** 2 / 100
    df['chol_squared'] = df['cholesterol_total'] ** 2 / 1000
    df['age_squared'] = df['age'] ** 2 / 1000
    
    return df

train_fe = advanced_feature_engineering(train)
test_fe = advanced_feature_engineering(test)

new_features = len(train_fe.columns) - len(train.columns)
print(f"✅ Created {new_features} new features")
print(f"✅ Total features: {len(train_fe.columns)}")

# ============================================================================
# STEP 3: Target Encoding (Simplified for speed)
# ============================================================================
print("\n🎯 STEP 3: Target Encoding...")
print("-" * 70)

from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin

class TargetEncoder(BaseEstimator, TransformerMixin):
    """Target Encoder with cross-validation"""
    def __init__(self, cols_to_encode, cv=5, smooth='auto'):
        self.cols_to_encode = cols_to_encode
        self.cv = cv
        self.smooth = smooth
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

# Prepare data
y = train_fe['diagnosed_diabetes']
X = train_fe.drop(columns=['diagnosed_diabetes', 'id'] if 'id' in train_fe.columns else ['diagnosed_diabetes'])
X_test = test_fe.drop(columns=['id'] if 'id' in test_fe.columns else [])

# Label encode categoricals
from sklearn.preprocessing import LabelEncoder
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([X[col], X_test[col]], axis=0).astype(str)
    le.fit(combined)
    X[col] = le.transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

# Identify integer columns for target encoding
int_cols = X.select_dtypes(include=['int64', 'int32']).columns.tolist()
int_cols = [col for col in int_cols if 'diagnosed' not in col.lower()]

print(f"✅ Prepared data: X={X.shape}, y={y.shape}, X_test={X_test.shape}")
print(f"✅ Integer columns for TE: {len(int_cols)}")

# ============================================================================
# STEP 4: Model Training with Cross-Validation
# ============================================================================
print("\n🤖 STEP 4: Model Training (5-Fold CV)...")
print("-" * 70)
print("⚠️  Note: Using smaller sample for faster execution")
print("    Full training would use all 700K samples")

# Use sample for faster execution (comment out for full training)
SAMPLE_SIZE = 50000  # Use 50K samples for faster testing
USE_SAMPLE = True

if USE_SAMPLE:
    print(f"📊 Using sample: {SAMPLE_SIZE} samples")
    X_sample = X.sample(n=min(SAMPLE_SIZE, len(X)), random_state=42)
    y_sample = y.loc[X_sample.index]
else:
    X_sample = X
    y_sample = y

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Arrays to store predictions
oof_preds_xgb = np.zeros(len(X_sample))
oof_preds_lgb = np.zeros(len(X_sample))
oof_preds_cat = np.zeros(len(X_sample))
test_preds_xgb = np.zeros(len(X_test))
test_preds_lgb = np.zeros(len(X_test))
test_preds_cat = np.zeros(len(X_test))

scores_xgb = []
scores_lgb = []
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
    
    # Model 1: XGBoost
    try:
        from xgboost import XGBClassifier
        xgb_model = XGBClassifier(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=5,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_lambda=1.0,
            reg_alpha=0.3,
            min_child_weight=3,
            tree_method='hist',
            random_state=42 + fold,
            eval_metric='auc',
            n_jobs=-1
        )
        xgb_model.fit(X_train_te, y_train, eval_set=[(X_val_te, y_val)], verbose=False, early_stopping_rounds=100)
        oof_preds_xgb[val_idx] = xgb_model.predict_proba(X_val_te)[:, 1]
        test_preds_xgb += xgb_model.predict_proba(X_test_te)[:, 1] / n_folds
        score_xgb = roc_auc_score(y_val, oof_preds_xgb[val_idx])
        scores_xgb.append(score_xgb)
        print(f"   XGBoost AUC: {score_xgb:.5f}")
    except Exception as e:
        print(f"   XGBoost: Error - {e}")
    
    # Model 2: LightGBM
    try:
        import lightgbm as lgb
        from lightgbm import LGBMClassifier
        lgb_model = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.01,
            num_leaves=50,
            max_depth=3,
            subsample=0.6,
            colsample_bytree=0.6,
            min_child_samples=50,
            reg_alpha=0.3,
            reg_lambda=1.0,
            random_state=42 + fold,
            metric='auc',
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(X_train_te, y_train, eval_set=[(X_val_te, y_val)], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        oof_preds_lgb[val_idx] = lgb_model.predict_proba(X_val_te)[:, 1]
        test_preds_lgb += lgb_model.predict_proba(X_test_te)[:, 1] / n_folds
        score_lgb = roc_auc_score(y_val, oof_preds_lgb[val_idx])
        scores_lgb.append(score_lgb)
        print(f"   LightGBM AUC: {score_lgb:.5f}")
    except Exception as e:
        print(f"   LightGBM: Error - {e}")
    
    # Model 3: CatBoost
    try:
        from catboost import CatBoostClassifier
        cat_model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.01,
            depth=3,
            l2_leaf_reg=3,
            bagging_temperature=1,
            random_seed=42 + fold,
            eval_metric='AUC',
            verbose=False,
            early_stopping_rounds=100
        )
        cat_model.fit(X_train_te, y_train, eval_set=(X_val_te, y_val), verbose=False)
        oof_preds_cat[val_idx] = cat_model.predict_proba(X_val_te)[:, 1]
        test_preds_cat += cat_model.predict_proba(X_test_te)[:, 1] / n_folds
        score_cat = roc_auc_score(y_val, oof_preds_cat[val_idx])
        scores_cat.append(score_cat)
        print(f"   CatBoost AUC: {score_cat:.5f}")
    except Exception as e:
        print(f"   CatBoost: Error - {e}")

elapsed_time = time.time() - start_time

# ============================================================================
# STEP 5: Results & Ensemble
# ============================================================================
print("\n" + "=" * 70)
print("📊 RESULTS")
print("=" * 70)

# Calculate CV scores
cv_xgb = roc_auc_score(y_sample, oof_preds_xgb) if len(scores_xgb) > 0 else 0
cv_lgb = roc_auc_score(y_sample, oof_preds_lgb) if len(scores_lgb) > 0 else 0
cv_cat = roc_auc_score(y_sample, oof_preds_cat) if len(scores_cat) > 0 else 0

# Ensemble
oof_ensemble = np.zeros(len(y_sample))
if cv_xgb > 0:
    oof_ensemble += oof_preds_xgb * (cv_xgb / (cv_xgb + cv_lgb + cv_cat) if (cv_xgb + cv_lgb + cv_cat) > 0 else 1/3)
if cv_lgb > 0:
    oof_ensemble += oof_preds_lgb * (cv_lgb / (cv_xgb + cv_lgb + cv_cat) if (cv_xgb + cv_lgb + cv_cat) > 0 else 1/3)
if cv_cat > 0:
    oof_ensemble += oof_preds_cat * (cv_cat / (cv_xgb + cv_lgb + cv_cat) if (cv_xgb + cv_lgb + cv_cat) > 0 else 1/3)

cv_ensemble = roc_auc_score(y_sample, oof_ensemble)

print(f"\nCross-Validation Scores ({'Sample' if USE_SAMPLE else 'Full'} Dataset):")
std_xgb = np.std(scores_xgb) if len(scores_xgb) > 0 else 0.0
std_lgb = np.std(scores_lgb) if len(scores_lgb) > 0 else 0.0
std_cat = np.std(scores_cat) if len(scores_cat) > 0 else 0.0
print(f"  XGBoost  CV AUC: {cv_xgb:.5f} (std: {std_xgb:.5f})")
print(f"  LightGBM CV AUC: {cv_lgb:.5f} (std: {std_lgb:.5f})")
print(f"  CatBoost  CV AUC: {cv_cat:.5f} (std: {std_cat:.5f})")
print(f"  {'-' * 60}")
print(f"  ENSEMBLE CV AUC: {cv_ensemble:.5f}")
print(f"\n⏱️  Training Time: {elapsed_time/60:.1f} minutes")

# Create ensemble test predictions
test_preds_ensemble = np.zeros(len(X_test))
weights_sum = 0
if cv_xgb > 0:
    weight_xgb = cv_xgb / (cv_xgb + cv_lgb + cv_cat) if (cv_xgb + cv_lgb + cv_cat) > 0 else 1/3
    test_preds_ensemble += test_preds_xgb * weight_xgb
    weights_sum += weight_xgb
if cv_lgb > 0:
    weight_lgb = cv_lgb / (cv_xgb + cv_lgb + cv_cat) if (cv_xgb + cv_lgb + cv_cat) > 0 else 1/3
    test_preds_ensemble += test_preds_lgb * weight_lgb
    weights_sum += weight_lgb
if cv_cat > 0:
    weight_cat = cv_cat / (cv_xgb + cv_lgb + cv_cat) if (cv_xgb + cv_lgb + cv_cat) > 0 else 1/3
    test_preds_ensemble += test_preds_cat * weight_cat
    weights_sum += weight_cat

if weights_sum > 0:
    test_preds_ensemble /= weights_sum

# ============================================================================
# STEP 6: Create Submission
# ============================================================================
print("\n💾 Creating Submission File...")
print("-" * 70)

submission = pd.DataFrame({
    'id': test['id'] if 'id' in test.columns else range(len(test)),
    'diagnosed_diabetes': test_preds_ensemble
})

submission_path = 'submission_comprehensive.csv'
submission.to_csv(submission_path, index=False)

print(f"✅ Submission saved: {submission_path}")
print(f"✅ Predictions range: [{test_preds_ensemble.min():.4f}, {test_preds_ensemble.max():.4f}]")
print(f"✅ Mean prediction: {test_preds_ensemble.mean():.4f}")

print("\n" + "=" * 70)
print("✅ PIPELINE COMPLETE!")
print("=" * 70)
print(f"\n📈 Expected Performance (on full dataset):")
print(f"   CV AUC: ~0.78-0.79")
print(f"   Public Score: ~0.70-0.72")
print(f"\n💡 Note: This run used {'sample' if USE_SAMPLE else 'full'} dataset")
print(f"   For best results, set USE_SAMPLE=False and run on full 700K samples")

