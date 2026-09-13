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

minimal_features = ['RET_1', 'MEDIAN_DAILY_TURNOVER', 'ALLOC_TARGET_ENCODING', 'ALLOC_VOLATILITY']

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

    # --- build BOTH allocation stats using ONLY this fold's training rows ---
    fold_train_targets = y_train.loc[local_train_ids, 'target']
    fold_train_allocs = X_train_minimal.loc[local_train_ids, 'ALLOCATION']
    fold_df = pd.DataFrame({'ALLOCATION': fold_train_allocs, 'target': fold_train_targets.values})

    alloc_sign_rate = fold_df.groupby('ALLOCATION')['target'].apply(lambda s: (np.sign(s) > 0).mean())
    alloc_volatility = fold_df.groupby('ALLOCATION')['target'].apply(lambda s: s.std())

    global_sign_rate = (np.sign(fold_train_targets) > 0).mean()
    global_volatility = fold_train_targets.std()

    for ids in [local_train_ids, local_test_ids]:
        X_train_minimal.loc[ids, 'ALLOC_TARGET_ENCODING'] = (
            X_train_minimal.loc[ids, 'ALLOCATION'].map(alloc_sign_rate).fillna(global_sign_rate)
        )
        X_train_minimal.loc[ids, 'ALLOC_VOLATILITY'] = (
            X_train_minimal.loc[ids, 'ALLOCATION'].map(alloc_volatility).fillna(global_volatility)
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
print(f"\nMinimal model accuracy: {mean:.2f}% (+- {std:.2f})")

# --- final model for submission: encode using FULL training set, no fold-holding needed
#     since X_test rows are genuinely unseen ---
full_alloc_sign_rate = total_data.groupby('ALLOCATION')['target'].apply(lambda s: (np.sign(s) > 0).mean())
full_alloc_volatility = total_data.groupby('ALLOCATION')['target'].apply(lambda s: s.std())
full_global_sign_rate = (np.sign(total_data['target']) > 0).mean()
full_global_volatility = total_data['target'].std()

X_train_minimal['ALLOC_TARGET_ENCODING_FULL'] = (
    X_train_minimal['ALLOCATION'].map(full_alloc_sign_rate).fillna(full_global_sign_rate)
)
X_train_minimal['ALLOC_VOLATILITY_FULL'] = (
    X_train_minimal['ALLOCATION'].map(full_alloc_volatility).fillna(full_global_volatility)
)

X_test = X_test.copy()
X_test['ALLOC_TARGET_ENCODING'] = X_test['ALLOCATION'].map(full_alloc_sign_rate).fillna(full_global_sign_rate)
X_test['ALLOC_VOLATILITY'] = X_test['ALLOCATION'].map(full_alloc_volatility).fillna(full_global_volatility)

final_train_df = X_train_minimal.rename(
    columns={'ALLOC_TARGET_ENCODING_FULL': 'ALLOC_TARGET_ENCODING', 'ALLOC_VOLATILITY_FULL': 'ALLOC_VOLATILITY'}
)
final_train_data = lgbm.Dataset(final_train_df[minimal_features], label=y_train['target'].values)
final_model = lgbm.train(lgbm_params, final_train_data, num_boost_round=NUM_BOOST_ROUND)

preds_lgbm = final_model.predict(X_test[minimal_features], num_threads=lgbm_params['num_threads'])
preds_lgbm = pd.DataFrame(preds_lgbm, index=sample_submission.index, columns=['target'])
(preds_lgbm > 0).astype(int).to_csv('outputs/predictions_lgbm_allocations_volatility.csv', index=True)