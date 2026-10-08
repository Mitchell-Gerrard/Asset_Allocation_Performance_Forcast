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

minimal_features = ['RET_1', 'MEDIAN_DAILY_TURNOVER', 'ALLOC_TARGET_ENCODING']

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

meta_lgbm_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_threads": 50,
    "seed": 42,
    "verbosity": -1,
    'learning_rate': 1e-2,
    'max_depth': 3,
}

NUM_BOOST_ROUND = 500
META_NUM_BOOST_ROUND = 300
N_INNER_SPLITS = 4  # for generating OOF primary predictions to train the meta-model on

train_dates = X_train_minimal['TS'].unique()
n_splits = 8
scores_minimal = []
scores_meta = []

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

    # --- PRIMARY MODEL: trained as before ---
    train_data = lgbm.Dataset(X_local_train, label=y_local_train.values)
    model = lgbm.train(lgbm_params, train_data, num_boost_round=NUM_BOOST_ROUND)
    y_local_pred = model.predict(X_local_test, num_threads=lgbm_params['num_threads'])

    score = accuracy_score((y_local_test > 0).astype(int), (y_local_pred > 0).astype(int))
    scores_minimal.append(score)
    print(f"Fold {i+1} - Primary accuracy: {score * 100:.2f}%")

    # --- META-LABELING STAGE ---
    # Step 1: generate OOF primary predictions on the TRAINING portion via an inner CV,
    #         so the meta-model learns from honest "would the primary have been right here"
    #         signals, not in-sample fit.
    inner_dates = local_train_dates.copy()
    inner_splits = KFold(n_splits=N_INNER_SPLITS, random_state=1, shuffle=True).split(inner_dates)

    oof_primary_pred = pd.Series(index=X_local_train.index, dtype=float)

    for inner_train_idx, inner_val_idx in inner_splits:
        inner_train_dates = inner_dates[inner_train_idx]
        inner_val_dates = inner_dates[inner_val_idx]

        inner_train_mask = X_train_minimal.loc[local_train_ids, 'TS'].isin(inner_train_dates)
        inner_val_mask = X_train_minimal.loc[local_train_ids, 'TS'].isin(inner_val_dates)

        inner_train_data = lgbm.Dataset(
            X_local_train[inner_train_mask.values], label=y_local_train[inner_train_mask.values].values
        )
        inner_model = lgbm.train(lgbm_params, inner_train_data, num_boost_round=NUM_BOOST_ROUND)

        inner_val_pred = inner_model.predict(X_local_train[inner_val_mask.values], num_threads=lgbm_params['num_threads'])
        oof_primary_pred.loc[X_local_train.index[inner_val_mask.values]] = inner_val_pred

    # label for the meta-model: was the OOF primary prediction correct?
    primary_was_correct = (
        (oof_primary_pred > 0).astype(int) == (y_local_train > 0).astype(int)
    ).astype(int)

    # meta-features: same base features + the primary prediction itself (its magnitude/confidence)
    meta_features_train = X_local_train.copy()
    meta_features_train['PRIMARY_PRED'] = oof_primary_pred

    meta_train_data = lgbm.Dataset(meta_features_train, label=primary_was_correct.values)
    meta_model = lgbm.train(meta_lgbm_params, meta_train_data, num_boost_round=META_NUM_BOOST_ROUND)

    # Step 2: apply primary + meta model to the actual outer-fold test set
    meta_features_test = X_local_test.copy()
    meta_features_test['PRIMARY_PRED'] = y_local_pred

    meta_confidence = meta_model.predict(meta_features_test, num_threads=meta_lgbm_params['num_threads'])

    # if meta-model is confident (>0.5 chance) the primary is WRONG, flip the prediction;
    # otherwise keep the primary's call
    primary_direction = (y_local_pred > 0).astype(int)
    flipped_direction = 1 - primary_direction
    final_direction = np.where(meta_confidence > 0.5, primary_direction, flipped_direction)
    # NOTE: this line is intentionally inverted below -- see explanation after the code block

    meta_score = accuracy_score((y_local_test > 0).astype(int), final_direction)
    scores_meta.append(meta_score)
    print(f"Fold {i+1} - Meta-labeled accuracy: {meta_score * 100:.2f}%")

mean = np.mean(scores_minimal) * 100
std = np.std(scores_minimal) * 100
print(f"\nPrimary model accuracy: {mean:.2f}% (+- {std:.2f})")

meta_mean = np.mean(scores_meta) * 100
meta_std = np.std(scores_meta) * 100
print(f"Meta-labeled model accuracy: {meta_mean:.2f}% (+- {meta_std:.2f})")
