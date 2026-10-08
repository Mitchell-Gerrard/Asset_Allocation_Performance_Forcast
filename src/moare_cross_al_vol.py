import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgbm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

X_train = pd.read_csv('data/X_train.csv', index_col='ROW_ID')
X_test = pd.read_csv('data/X_test.csv', index_col='ROW_ID')
y_train = pd.read_csv('data/y_train.csv', index_col='ROW_ID')
sample_submission = pd.read_csv('data/sample_submission.csv', index_col='ROW_ID')
total_data = pd.concat([X_train, y_train], axis=1)

RET_features = [f'RET_{i}' for i in range(1, 21)]
SIGNED_VOLUME_features = [f'SIGNED_VOLUME_{i}' for i in range(1, 21)]
TURNOVER_features = ['MEDIAN_DAILY_TURNOVER']
X_train['CROSS_SECTIONAL_AVG_RET1'] = total_data.groupby('TS')['RET_1'].transform('mean')
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

features = RET_features + SIGNED_VOLUME_features + TURNOVER_features
features = features + [f'AVERAGE_PERF_{i}' for i in [3, 5, 10, 15, 20]]
features = features + [f'ALLOCATIONS_AVERAGE_PERF_{i}' for i in [3, 5, 10, 15, 20]]
features = features + [f'STD_PERF_{i}' for i in [20]]
features = features + [f'ALLOCATIONS_STD_PERF_{i}' for i in [20]]
minimal_features = features + ['CROSS_SECTIONAL_AVG_RET1', 'ALLOC_TARGET_ENCODING']
SIGNED_VOLUME_features_clean = [f'SIGNED_VOLUME_{i}' for i in range(2, 21)]

for i in [3, 5, 10, 15, 20]:
    cols = SIGNED_VOLUME_features_clean[:i-1]  # i-1 since we're starting from lag 2, not lag 1
    X_train[f'AVERAGE_VOLUME_{i}'] = X_train[cols].mean(1)
    X_train[f'ALLOCATIONS_AVERAGE_VOLUME_{i}'] = X_train.groupby('TS')[f'AVERAGE_VOLUME_{i}'].transform('mean')
    X_test[f'AVERAGE_VOLUME_{i}'] = X_test[cols].mean(1)
    X_test[f'ALLOCATIONS_AVERAGE_VOLUME_{i}'] = X_test.groupby('TS')[f'AVERAGE_VOLUME_{i}'].transform('mean')

for i in [20]:
    cols = SIGNED_VOLUME_features_clean[:i-1]
    X_train[f'STD_VOLUME_{i}'] = X_train[cols].std(1)
    X_train[f'ALLOCATIONS_STD_VOLUME_{i}'] = X_train.groupby('TS')[f'STD_VOLUME_{i}'].transform('mean')
    X_test[f'STD_VOLUME_{i}'] = X_test[cols].std(1)
    X_test[f'ALLOCATIONS_STD_VOLUME_{i}'] = X_test.groupby('TS')[f'STD_VOLUME_{i}'].transform('mean')

volume_features = (
    [f'AVERAGE_VOLUME_{i}' for i in [3, 5, 10, 15, 20]]
    + [f'ALLOCATIONS_AVERAGE_VOLUME_{i}' for i in [3, 5, 10, 15, 20]]
    + [f'STD_VOLUME_{i}' for i in [20]]
    + [f'ALLOCATIONS_STD_VOLUME_{i}' for i in [20]]
)
features = features + volume_features
minimal_features = features + ['CROSS_SECTIONAL_AVG_RET1', 'ALLOC_TARGET_ENCODING']
X_train_minimal = X_train.copy()

lgbm_params = {
    "objective": "mse",
    "metric": "mse",
    "num_threads": 50,
    "seed": 42,
    "verbosity": -1,
    'learning_rate': 1e-2,
    'max_depth': 3,
}
NUM_BOOST_ROUND = 500

train_dates = X_train_minimal['TS'].unique()
n_splits = 8
scores_minimal = []

splits = KFold(n_splits=n_splits, random_state=0, shuffle=True).split(train_dates)

for i, (local_train_dates_ids, local_test_dates_ids) in enumerate(splits):
    local_train_dates = train_dates[local_train_dates_ids]
    local_test_dates = train_dates[local_test_dates_ids]

    local_train_ids = X_train_minimal['TS'].isin(local_train_dates)
    local_test_ids = X_train_minimal['TS'].isin(local_test_dates)

    # --- build the allocation target encoding using ONLY this fold's training rows ---
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

    train_data = lgbm.Dataset(X_local_train, label=y_local_train.values)

    model = lgbm.train(lgbm_params, train_data, num_boost_round=NUM_BOOST_ROUND)
    y_local_pred = model.predict(X_local_test, num_threads=lgbm_params['num_threads'])

    score = accuracy_score((y_local_test > 0).astype(int), (y_local_pred > 0).astype(int))
    scores_minimal.append(score)
    print(f"Fold {i+1} - Accuracy: {score * 100:.2f}%")

mean = np.mean(scores_minimal) * 100
std = np.std(scores_minimal) * 100
print(f"\nCombined model accuracy: {mean:.2f}% (+- {std:.2f})")

# --- final model for submission: encode using the FULL training set ---
full_alloc_encoding = (
    total_data.groupby('ALLOCATION')['target']
    .apply(lambda s: (np.sign(s) > 0).mean())
)
full_global_rate = (np.sign(total_data['target']) > 0).mean()

X_test['ALLOC_TARGET_ENCODING'] = X_test['ALLOCATION'].map(full_alloc_encoding).fillna(full_global_rate)
X_test['CROSS_SECTIONAL_AVG_RET1'] = X_test.groupby('TS')['RET_1'].transform('mean')
X_train_minimal['ALLOC_TARGET_ENCODING'] = (
    X_train_minimal['ALLOCATION'].map(full_alloc_encoding).fillna(full_global_rate)
)

final_train_data = lgbm.Dataset(X_train_minimal[minimal_features], label=y_train['target'].values)
final_model = lgbm.train(lgbm_params, final_train_data, num_boost_round=NUM_BOOST_ROUND)

preds_lgbm = final_model.predict(X_test[minimal_features], num_threads=lgbm_params['num_threads'])
preds_lgbm = pd.DataFrame(preds_lgbm, index=sample_submission.index, columns=['target'])
(preds_lgbm > 0).astype(int).to_csv('outputs/predictions_lgbm_allocations_cross_volume_plus_baseline.csv', index=True)