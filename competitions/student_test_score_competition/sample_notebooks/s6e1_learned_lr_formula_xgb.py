import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost==2.0.0"])

subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.26.4"])
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import xgboost as xgb
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
train_file = "/kaggle/input/playground-series-s6e1/train.csv"
test_file = "/kaggle/input/playground-series-s6e1/test.csv"
original_file = "/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv"
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
original_df = pd.read_csv(original_file)
submission_df = pd.read_csv("/kaggle/input/playground-series-s6e1/sample_submission.csv")
TARGET = 'exam_score'
base_features = [col for col in train_df.columns if col not in [TARGET, 'id']]

# %% [code] {"execution":{"iopub.status.busy":"2026-01-02T11:30:59.16623Z","iopub.execute_input":"2026-01-02T11:30:59.166816Z","iopub.status.idle":"2026-01-02T11:30:59.21426Z","shell.execute_reply.started":"2026-01-02T11:30:59.166789Z","shell.execute_reply":"2026-01-02T11:30:59.213576Z"}}

CATS = train_df.select_dtypes('object').columns.to_list()

# %% [code] {"execution":{"iopub.status.busy":"2026-01-02T11:30:41.795967Z","iopub.execute_input":"2026-01-02T11:30:41.796245Z","iopub.status.idle":"2026-01-02T11:30:41.80261Z","shell.execute_reply.started":"2026-01-02T11:30:41.796225Z","shell.execute_reply":"2026-01-02T11:30:41.80193Z"}}
def preprocess(df):
    df_temp = df.copy()

    # instead of this constant, we use a LR to learn the linear part
    # df_temp['feature_formula'] = (
    #     5.9051154511950499 * df_temp['study_hours'] +
    #     0.34540967058057986 * df_temp['class_attendance'] +
    #     1.423461171860262 * df_temp['sleep_hours'] + 4.7819
    # )

    df_temp['study_hours_squared'] = df_temp['study_hours'] ** 2
    df_temp['study_hours_cubed'] = df_temp['study_hours'] ** 3
    df_temp['class_attendance_squared'] = df_temp['class_attendance'] ** 2
    df_temp['sleep_hours_squared'] = df_temp['sleep_hours'] ** 2
    df_temp['age_squared'] = df_temp['age'] ** 2

    df_temp['log_study_hours'] = np.log1p(df_temp['study_hours'])
    df_temp['log_class_attendance'] = np.log1p(df_temp['class_attendance'])
    df_temp['log_sleep_hours'] = np.log1p(df_temp['sleep_hours'])
    df_temp['sqrt_study_hours'] = np.sqrt(df_temp['study_hours'])
    df_temp['sqrt_class_attendance'] = np.sqrt(df_temp['class_attendance'])

    # Interaction features
    df_temp['study_hours_times_attendance'] = df_temp['study_hours'] * df_temp['class_attendance']
    df_temp['study_hours_times_sleep'] = df_temp['study_hours'] * df_temp['sleep_hours']
    df_temp['attendance_times_sleep'] = df_temp['class_attendance'] * df_temp['sleep_hours']

    # Ratio features (add small epsilon to avoid division by zero)
    eps = 1e-5
    df_temp['study_hours_over_sleep'] = df_temp['study_hours'] / (df_temp['sleep_hours'] + eps)
    df_temp['attendance_over_sleep'] = df_temp['class_attendance'] / (df_temp['sleep_hours'] + eps)

    # Encode categorical variables to numeric ordinal values
    sleep_quality_map = {'poor': 0, 'average': 1, 'good': 2}
    facility_rating_map = {'low': 0, 'medium': 1, 'high': 2}
    exam_difficulty_map = {'easy': 0, 'medium': 1, 'hard': 2}

    df_temp['sleep_quality_numeric'] = df_temp['sleep_quality'].map(sleep_quality_map).fillna(1).astype(int)
    df_temp['facility_rating_numeric'] = df_temp['facility_rating'].map(facility_rating_map).fillna(1).astype(int)
    df_temp['exam_difficulty_numeric'] = df_temp['exam_difficulty'].map(exam_difficulty_map).fillna(1).astype(int)

    # Interaction between encoded categoricals and key numeric features
    df_temp['study_hours_times_sleep_quality'] = df_temp['study_hours'] * df_temp['sleep_quality_numeric']
    df_temp['attendance_times_facility'] = df_temp['class_attendance'] * df_temp['facility_rating_numeric']
    df_temp['sleep_hours_times_difficulty'] = df_temp['sleep_hours'] * df_temp['exam_difficulty_numeric']
    df_temp['age_times_study_hours'] = df_temp['age'] * df_temp['study_hours']
    df_temp['age_times_attendance'] = df_temp['age'] * df_temp['class_attendance']

    # Composite feature: learning efficiency
    df_temp['efficiency'] = (df_temp['study_hours'] * df_temp['class_attendance']) / (df_temp['sleep_hours'] + 1)

    numeric_features = [
        # 'feature_formula',
        'study_hours_squared', 'study_hours_cubed',
        'class_attendance_squared', 'sleep_hours_squared', 'age_squared',
        'log_study_hours', 'log_class_attendance', 'log_sleep_hours',
        'sqrt_study_hours', 'sqrt_class_attendance',
        'study_hours_times_attendance', 'study_hours_times_sleep',
        'attendance_times_sleep', 'study_hours_over_sleep',
        'attendance_over_sleep',
        'sleep_quality_numeric', 'facility_rating_numeric', 'exam_difficulty_numeric',
        'study_hours_times_sleep_quality', 'attendance_times_facility',
        'sleep_hours_times_difficulty', 'age_times_study_hours',
        'age_times_attendance', 'efficiency']

    return df_temp[base_features + numeric_features]

# %% [code] {"execution":{"iopub.status.busy":"2026-01-02T11:30:42.547299Z","iopub.execute_input":"2026-01-02T11:30:42.547816Z","iopub.status.idle":"2026-01-02T11:30:43.242268Z","shell.execute_reply.started":"2026-01-02T11:30:42.54779Z","shell.execute_reply":"2026-01-02T11:30:43.241481Z"}}
X_raw = preprocess(train_df)
y = train_df[TARGET].reset_index(drop=True)

X_test_raw = preprocess(test_df)
X_orig_raw = preprocess(original_df)
y_orig = original_df[TARGET].reset_index(drop=True)

full_data = pd.concat([X_raw, X_test_raw, X_orig_raw], axis=0)

# LR doesn't use categorical
# for col in base_features:
#     df_temp[col] = df_temp[col].astype(str)
# for col in base_features:
#     full_data[col] = full_data[col].astype('category')

numeric_cols = [
    # 'feature_formula',
    'study_hours_squared', 'study_hours_cubed',
    'class_attendance_squared', 'sleep_hours_squared', 'age_squared',
    'log_study_hours', 'log_class_attendance', 'log_sleep_hours',
    'sqrt_study_hours', 'sqrt_class_attendance',
    'study_hours_times_attendance', 'study_hours_times_sleep',
    'attendance_times_sleep', 'study_hours_over_sleep',
    'attendance_over_sleep',
    'sleep_quality_numeric', 'facility_rating_numeric', 'exam_difficulty_numeric',
    'study_hours_times_sleep_quality', 'attendance_times_facility',
    'sleep_hours_times_difficulty', 'age_times_study_hours',
    'age_times_attendance', 'efficiency']
for col in numeric_cols:
    full_data[col] = full_data[col].astype(float)

X = full_data.iloc[:len(train_df)].copy()
X_test = full_data.iloc[len(train_df):len(train_df) + len(test_df)].copy()
X_original = full_data.iloc[len(train_df) + len(test_df):].copy()

# %% [markdown]
# # Learn the linear part

# %% [code] {"execution":{"iopub.status.busy":"2026-01-02T11:30:45.567905Z","iopub.execute_input":"2026-01-02T11:30:45.56818Z","iopub.status.idle":"2026-01-02T11:30:45.6536Z","shell.execute_reply.started":"2026-01-02T11:30:45.568159Z","shell.execute_reply":"2026-01-02T11:30:45.652861Z"}}
from sklearn.linear_model import RidgeCV
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import TargetEncoder

# %% [code] {"execution":{"iopub.status.busy":"2026-01-02T11:31:01.39429Z","iopub.execute_input":"2026-01-02T11:31:01.39458Z","iopub.status.idle":"2026-01-02T11:31:12.198968Z","shell.execute_reply.started":"2026-01-02T11:31:01.394555Z","shell.execute_reply":"2026-01-02T11:31:12.195929Z"}}
N_SAMPLES_TRAIN = X.shape[0]
N_SAMPLES_TEST = X_test.shape[0]
FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=1003)

oof_pred_lr = np.zeros(N_SAMPLES_TRAIN)
test_preds_lr = np.zeros((N_SAMPLES_TEST, FOLDS))
orig_preds_lr = np.zeros(X_original.shape[0])
fold_rmse_lr = []
lr_models = []

for fold, (train_index, val_index) in enumerate(kf.split(X, y), start=1):
    print(f"Training fold {fold} ...")

    X_train_fold, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train_fold, y_val = y.iloc[train_index], y.iloc[val_index]

    X_train_combined = pd.concat([X_train_fold, X_original], axis=0)
    y_train_combined = pd.concat([y_train_fold, y_orig], axis=0)

    # target encode categorical features
    target_encoder = TargetEncoder(smooth='auto', target_type='continuous')
    X_train_encoded = X_train_combined.copy()
    X_val_encoded = X_val.copy()
    X_test_encoded = X_test.copy()

    X_train_encoded[CATS] = target_encoder.fit_transform(X_train_combined[CATS], y_train_combined)
    X_val_encoded[CATS] = target_encoder.transform(X_val[CATS])
    X_test_encoded[CATS] = target_encoder.transform(X_test[CATS])

    # train regularized linear regression model with cross-validated alpha selection
    # RidgeCV with alphas log-spaced for better coverage
    alphas = np.logspace(-3, 3, 20)  # 20 values from 0.001 to 1000
    lr_model = RidgeCV(alphas=alphas, cv=5, scoring='neg_root_mean_squared_error')
    lr_model.fit(X_train_encoded, y_train_combined.ravel())
    lr_models.append(lr_model)
    print(f"Fold {fold} selected alpha: {lr_model.alpha_:.4f}")

    # compute OOF predictions/test predictions
    lr_val_pred = lr_model.predict(X_val_encoded)
    lr_test_pred = lr_model.predict(X_test_encoded)
    lr_orig_pred = lr_model.predict(X_train_encoded.iloc[-X_original.shape[0]:])

    # clip values
    lr_val_pred = np.clip(lr_val_pred, 0, 100)
    lr_test_pred = np.clip(lr_test_pred, 0, 100)
    lr_orig_pred = np.clip(lr_orig_pred, 0, 100)

    # evaluate
    rmse_lr = root_mean_squared_error(y_val, lr_val_pred)
    oof_pred_lr[val_index] = lr_val_pred
    test_preds_lr[:, fold - 1] = lr_test_pred
    orig_preds_lr += lr_orig_pred / FOLDS

    print(f"Fold {fold} RMSE (lr): {rmse_lr:.8f}")
    fold_rmse_lr.append(rmse_lr)

# %% [markdown]
# # XGB categorical

# %% [code] {"execution":{"iopub.status.busy":"2026-01-02T11:31:12.199899Z","iopub.execute_input":"2026-01-02T11:31:12.200123Z","iopub.status.idle":"2026-01-02T11:31:14.485643Z","shell.execute_reply.started":"2026-01-02T11:31:12.200101Z","shell.execute_reply":"2026-01-02T11:31:14.484885Z"}}
# for XGB - treat rest as categories
for col in base_features:
    full_data[col] = full_data[col].astype(str)
    full_data[col] = full_data[col].astype('category')

for col in numeric_cols:
    full_data[col] = full_data[col].astype(float)

X = full_data.iloc[:len(train_df)].copy()
X_test = full_data.iloc[len(train_df):len(train_df) + len(test_df)].copy()
X_original = full_data.iloc[len(train_df) + len(test_df):].copy()

# add predictions on original data as features
X['feature_lr_pred'] = oof_pred_lr
X_test['feature_lr_pred'] = test_preds_lr.mean(axis=1)
X_original['feature_lr_pred'] = orig_preds_lr

# %% [code] {"execution":{"iopub.status.busy":"2026-01-02T11:31:19.422736Z","iopub.execute_input":"2026-01-02T11:31:19.423002Z","iopub.status.idle":"2026-01-02T11:31:19.428179Z","shell.execute_reply.started":"2026-01-02T11:31:19.422982Z","shell.execute_reply":"2026-01-02T11:31:19.427457Z"}}
xgb_params = {
    'n_estimators': 15000,
    'learning_rate': 0.005,
    'max_depth': 9,
    'subsample': 0.75,
    'reg_lambda': 5,
    'reg_alpha': 0.1,
    'colsample_bytree': 0.5,
    'colsample_bynode': 0.6,
    'min_child_weight': 5,
    'tree_method': 'hist',
    'random_state': 42,
    'early_stopping_rounds': 80,
    'eval_metric': 'rmse',
    'enable_categorical': True,
    'device': 'cuda'
}

test_predictions = []
oof_predictions = np.zeros(len(X))


# %% [code] {"execution":{"iopub.status.busy":"2026-01-02T11:31:20.926029Z","iopub.execute_input":"2026-01-02T11:31:20.926553Z","iopub.status.idle":"2026-01-02T11:31:45.225439Z","shell.execute_reply.started":"2026-01-02T11:31:20.926529Z","shell.execute_reply":"2026-01-02T11:31:45.224529Z"}}
for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
    print(f"\n--- Fold {fold + 1} ---")

    X_train_fold, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train_fold, y_val = y.iloc[train_index], y.iloc[val_index]

    X_train_combined = pd.concat([X_train_fold, X_original], axis=0)
    y_train_combined = pd.concat([y_train_fold, y_orig], axis=0)

    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train_combined, y_train_combined, eval_set=[(X_val, y_val)], verbose=1000)

    val_preds = model.predict(X_val)
    oof_predictions[val_index] = val_preds
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"RMSE: {rmse:.5f}")

    test_preds = model.predict(X_test)
    test_predictions.append(test_preds)

oof_rmse = np.sqrt(mean_squared_error(y, oof_predictions))

print("\n-----------------------")
print(f"OOF RMSE: {oof_rmse:.5f}")
print(f"Improvement from Step 2: {8.64444 - oof_rmse:.5f}")

oof_df = pd.DataFrame({'id': train_df['id'], TARGET: oof_predictions})
oof_df.to_csv('xgb_oof.csv', index=False)

submission_df[TARGET] = np.mean(test_predictions, axis=0)
submission_df.to_csv('submission.csv', index=False)
submission_df.head()
