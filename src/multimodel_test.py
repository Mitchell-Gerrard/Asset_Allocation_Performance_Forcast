import numpy as np
import pandas as pd
import lightgbm as lgbm
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# ============================================================
# DATA LOADING
# ============================================================
X_train = pd.read_csv('data/X_train.csv', index_col='ROW_ID')
X_test = pd.read_csv('data/X_test.csv', index_col='ROW_ID')
y_train = pd.read_csv('data/y_train.csv', index_col='ROW_ID')
sample_submission = pd.read_csv('data/sample_submission.csv', index_col='ROW_ID')
total_data = pd.concat([X_train, y_train], axis=1)

# ============================================================
# FEATURE ENGINEERING (your current best combined set)
# ============================================================
RET_features = [f'RET_{i}' for i in range(1, 21)]
SIGNED_VOLUME_features = [f'SIGNED_VOLUME_{i}' for i in range(1, 21)]
TURNOVER_features = ['MEDIAN_DAILY_TURNOVER']

X_train['CROSS_SECTIONAL_AVG_RET1'] = X_train.groupby('TS')['RET_1'].transform('mean')
X_test['CROSS_SECTIONAL_AVG_RET1'] = X_test.groupby('TS')['RET_1'].transform('mean')

for i in [3, 5, 10, 15, 20]:
    X_train[f'AVERAGE_PERF_{i}'] = X_train[RET_features[:i]].mean(1)
    X_train[f'ALLOCATIONS_AVERAGE_PERF_{i}'] = X_train.groupby('TS')[f'AVERAGE_PERF_{i}'].transform('mean')
    X_test[f'AVERAGE_PERF_{i}'] = X_test[RET_features[:i]].mean(1)
    X_test[f'ALLOCATIONS_AVERAGE_PERF_{i}'] = X_test.groupby('TS')[f'AVERAGE_PERF_{i}'].transform('mean')

for i in [20]:
    X_train[f'STD_PERF_{i}'] = X_train[RET_features[:i]].std(1)
    X_train[f'ALLOCATIONS_STD_PERF_{i}'] = X_train.groupby('TS')[f'STD_PERF_{i}'].transform('mean')
    X_test[f'STD_PERF_{i}'] = X_test[RET_features[:i]].std(1)
    X_test[f'ALLOCATIONS_STD_PERF_{i}'] = X_test.groupby('TS')[f'STD_PERF_{i}'].transform('mean')

SIGNED_VOLUME_features_clean = [f'SIGNED_VOLUME_{i}' for i in range(2, 21)]
for i in [3, 5, 10, 15, 20]:
    cols = SIGNED_VOLUME_features_clean[:i - 1]
    X_train[f'AVERAGE_VOLUME_{i}'] = X_train[cols].mean(1)
    X_train[f'ALLOCATIONS_AVERAGE_VOLUME_{i}'] = X_train.groupby('TS')[f'AVERAGE_VOLUME_{i}'].transform('mean')
    X_test[f'AVERAGE_VOLUME_{i}'] = X_test[cols].mean(1)
    X_test[f'ALLOCATIONS_AVERAGE_VOLUME_{i}'] = X_test.groupby('TS')[f'AVERAGE_VOLUME_{i}'].transform('mean')

for i in [20]:
    cols = SIGNED_VOLUME_features_clean[:i - 1]
    X_train[f'STD_VOLUME_{i}'] = X_train[cols].std(1)
    X_train[f'ALLOCATIONS_STD_VOLUME_{i}'] = X_train.groupby('TS')[f'STD_VOLUME_{i}'].transform('mean')
    X_test[f'STD_VOLUME_{i}'] = X_test[cols].std(1)
    X_test[f'ALLOCATIONS_STD_VOLUME_{i}'] = X_test.groupby('TS')[f'STD_VOLUME_{i}'].transform('mean')

features = RET_features + SIGNED_VOLUME_features + TURNOVER_features
features += [f'AVERAGE_PERF_{i}' for i in [3, 5, 10, 15, 20]]
features += [f'ALLOCATIONS_AVERAGE_PERF_{i}' for i in [3, 5, 10, 15, 20]]
features += [f'STD_PERF_{i}' for i in [20]]
features += [f'ALLOCATIONS_STD_PERF_{i}' for i in [20]]
features += [f'AVERAGE_VOLUME_{i}' for i in [3, 5, 10, 15, 20]]
features += [f'ALLOCATIONS_AVERAGE_VOLUME_{i}' for i in [3, 5, 10, 15, 20]]
features += [f'STD_VOLUME_{i}' for i in [20]]
features += [f'ALLOCATIONS_STD_VOLUME_{i}' for i in [20]]

minimal_features = features + ['CROSS_SECTIONAL_AVG_RET1', 'ALLOC_TARGET_ENCODING']

X_train_minimal = X_train.copy()

# ============================================================
# MODEL REGISTRY
# To add a new model type later: write a train_fn(X, y) -> fitted model,
# and a predict_fn(model, X) -> array of continuous scores (same sign
# convention as target -- positive means "predict long"). Add both to
# MODEL_REGISTRY below with a chosen ensemble weight.
# ============================================================

def train_lgbm(X, y):
    lgbm_params = {
        "objective": "mse",
        "metric": "mse",
        "num_threads": 50,
        "seed": 42,
        "verbosity": -1,
        'learning_rate': 1e-2,
        'max_depth': 3,
    }
    train_data = lgbm.Dataset(X, label=y.values)
    return lgbm.train(lgbm_params, train_data, num_boost_round=500)

def predict_lgbm(model, X):
    return model.predict(X, num_threads=50)

def train_ridge(X, y):
    # Ridge can't handle NaNs the way LightGBM can -- fill with 0 for this model only.
    model = Ridge(alpha=1e-2, fit_intercept=True)
    model.fit(X.fillna(0), y)
    return model

def predict_ridge(model, X):
    return model.predict(X.fillna(0))

# name -> (train_fn, predict_fn, ensemble_weight)
# Weights don't need to sum to 1 -- they're relative; predictions are combined as a
# weighted average. Set a model's weight to 0 to disable it without deleting code.
MODEL_REGISTRY = {
    'lgbm': (train_lgbm, predict_lgbm, 1.0),
    'ridge': (train_ridge, predict_ridge, 1.0),
    # add more here, e.g.:
    # 'xgboost': (train_xgb, predict_xgb, 1.0),
}

def train_and_predict_ensemble(X_local_train, y_local_train, X_local_test):
    """Trains every model in MODEL_REGISTRY with nonzero weight, returns the
    weighted-average continuous prediction plus each individual model's
    prediction (useful for per-model diagnostics)."""
    individual_preds = {}
    weighted_sum = np.zeros(len(X_local_test))
    total_weight = 0.0

    for name, (train_fn, predict_fn, weight) in MODEL_REGISTRY.items():
        if weight == 0:
            continue
        model = train_fn(X_local_train, y_local_train)
        pred = predict_fn(model, X_local_test)
        individual_preds[name] = pred
        weighted_sum += weight * pred
        total_weight += weight

    ensemble_pred = weighted_sum / total_weight
    return ensemble_pred, individual_preds

# ============================================================
# CROSS-VALIDATION LOOP
# ============================================================
train_dates = X_train_minimal['TS'].unique()
n_splits = 8

scores_ensemble = []
scores_by_model = {name: [] for name in MODEL_REGISTRY}

splits = KFold(n_splits=n_splits, random_state=0, shuffle=True).split(train_dates)

for i, (local_train_dates_ids, local_test_dates_ids) in enumerate(splits):
    local_train_dates = train_dates[local_train_dates_ids]
    local_test_dates = train_dates[local_test_dates_ids]

    local_train_ids = X_train_minimal['TS'].isin(local_train_dates)
    local_test_ids = X_train_minimal['TS'].isin(local_test_dates)

    # --- allocation target encoding, fold-scoped as before ---
    fold_train_targets = y_train.loc[local_train_ids, 'target']
    fold_train_allocs = X_train_minimal.loc[local_train_ids, 'ALLOCATION']

    alloc_encoding = (
        pd.DataFrame({'ALLOCATION': fold_train_allocs, 'target': fold_train_targets.values})
        .groupby('ALLOCATION')['target']
        .apply(lambda s: (np.sign(s) > 0).mean())
    )
    global_rate = (np.sign(fold_train_targets) > 0).mean()

    X_train_minimal.loc[local_train_ids, 'ALLOC_TARGET_ENCODING'] = (
        X_train_minimal.loc[local_train_ids, 'ALLOCATION'].map(alloc_encoding).fillna(global_rate)
    )
    X_train_minimal.loc[local_test_ids, 'ALLOC_TARGET_ENCODING'] = (
        X_train_minimal.loc[local_test_ids, 'ALLOCATION'].map(alloc_encoding).fillna(global_rate)
    )
    # --- end encoding block ---

    X_local_train = X_train_minimal.loc[local_train_ids, minimal_features]
    y_local_train = y_train.loc[local_train_ids, 'target']

    X_local_test = X_train_minimal.loc[local_test_ids, minimal_features]
    y_local_test = y_train.loc[local_test_ids, 'target']

    ensemble_pred, individual_preds = train_and_predict_ensemble(X_local_train, y_local_train, X_local_test)

    score = accuracy_score((y_local_test > 0).astype(int), (ensemble_pred > 0).astype(int))
    scores_ensemble.append(score)

    for name, pred in individual_preds.items():
        s = accuracy_score((y_local_test > 0).astype(int), (pred > 0).astype(int))
        scores_by_model[name].append(s)

    print(f"Fold {i+1} - Ensemble accuracy: {score * 100:.2f}%  |  " +
          "  ".join(f"{name}: {scores_by_model[name][-1] * 100:.2f}%" for name in MODEL_REGISTRY if MODEL_REGISTRY[name][2] != 0))

print(f"\nEnsemble accuracy: {np.mean(scores_ensemble) * 100:.2f}% (+- {np.std(scores_ensemble) * 100:.2f})")
for name in MODEL_REGISTRY:
    if MODEL_REGISTRY[name][2] == 0:
        continue
    s = scores_by_model[name]
    print(f"{name} accuracy: {np.mean(s) * 100:.2f}% (+- {np.std(s) * 100:.2f})")

# ============================================================
# FINAL MODEL FOR SUBMISSION (trained on full training set)
# ============================================================
full_alloc_encoding = (
    total_data.groupby('ALLOCATION')['target']
    .apply(lambda s: (np.sign(s) > 0).mean())
)
full_global_rate = (np.sign(total_data['target']) > 0).mean()

X_test['ALLOC_TARGET_ENCODING'] = X_test['ALLOCATION'].map(full_alloc_encoding).fillna(full_global_rate)
X_train_minimal['ALLOC_TARGET_ENCODING'] = (
    X_train_minimal['ALLOCATION'].map(full_alloc_encoding).fillna(full_global_rate)
)

final_ensemble_pred, final_individual_preds = train_and_predict_ensemble(
    X_train_minimal[minimal_features], y_train['target'], X_test[minimal_features]
)

preds_df = pd.DataFrame(final_ensemble_pred, index=sample_submission.index, columns=['target'])
(preds_df > 0).astype(int).to_csv('outputs/predictions_ensemble.csv', index=True)

# also save each individual model's predictions, useful for later blending experiments
for name, pred in final_individual_preds.items():
    individual_df = pd.DataFrame(pred, index=sample_submission.index, columns=['target'])
    (individual_df > 0).astype(int).to_csv(f'outputs/predictions_{name}_only.csv', index=True)