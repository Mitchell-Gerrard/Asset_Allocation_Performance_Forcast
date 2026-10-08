import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgbm
import scipy
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.calibration import calibration_curve
from sklearn.model_selection import TimeSeriesSplit

import shap
import scipy.stats as stats
def add_cross_sectional_zscore(X, base_features, suffix='_ZSCORE'):
    """
    For each feature in base_features, compute its z-score relative to the
    cross-section of all allocations present at the same TS.
    z = (x - mean(x at this TS)) / std(x at this TS)

    This isolates relative strength on a given day, independent of that
    day's overall market regime (volatile vs calm days), which raw values
    conflate.
    """
    new_cols = []
    for feat in base_features:
        ts_mean = X.groupby('TS')[feat].transform('mean')
        ts_std = X.groupby('TS')[feat].transform('std')

        new_col = f'{feat}{suffix}'
        # avoid division by zero on days where the feature is constant across the universe
        X[new_col] = np.where(ts_std > 0, (X[feat] - ts_mean) / ts_std, 0.0)
        new_cols.append(new_col)

    return X, new_cols
def compute_bayesian_alloc_rate(train_targets, train_allocs, prior_strength=50):
    """
    Beta-Binomial posterior mean for each allocation's positive-rate.
    prior_strength (alpha0 + beta0) controls how much weight the prior carries,
    same role as 'k' in your earlier shrinkage estimator -- but here it's an
    explicit, principled Bayesian prior, not an ad-hoc formula.
    """
    global_rate = (np.sign(train_targets) > 0).mean()

    # prior: Beta(alpha0, beta0) centered on the global rate, with `prior_strength`
    # total pseudo-observations
    alpha0 = global_rate * prior_strength
    beta0 = (1 - global_rate) * prior_strength

    df = pd.DataFrame({'ALLOCATION': train_allocs, 'target': train_targets.values})
    grouped = df.groupby('ALLOCATION')['target']

    successes = grouped.apply(lambda s: (np.sign(s) > 0).sum())
    n = grouped.size()

    # posterior: Beta(alpha0 + successes, beta0 + failures)
    posterior_alpha = alpha0 + successes
    posterior_beta = beta0 + (n - successes)

    posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)

    # bonus: a genuine 90% credible interval per allocation, not just a point estimate
    ci_lower = stats.beta.ppf(0.05, posterior_alpha, posterior_beta)
    ci_upper = stats.beta.ppf(0.95, posterior_alpha, posterior_beta)

    return posterior_mean, ci_lower, ci_upper, global_rate
X_train = pd.read_csv('data/X_train.csv', index_col='ROW_ID')
X_test = pd.read_csv('data/X_test.csv', index_col='ROW_ID')
y_train = pd.read_csv('data/y_train.csv', index_col='ROW_ID')
y_train_bin = (y_train['target'] > 0).astype(int)
sample_submission = pd.read_csv('data/sample_submission.csv', index_col='ROW_ID')
total_data = pd.concat([X_train, y_train_bin.rename('target')], axis=1)
ret_cols = [col for col in total_data.columns if 'RET' in col]
for X in [X_train, X_test]:
    for i, col in enumerate(ret_cols):
        if i == 0:
            continue
        momentum_col = f'MOMENTUM_{i}'
        X[momentum_col] = (X["RET_1"] / X[col]) - 1
momentum_cols = [col for col in X_train.columns if 'MOMENTUM' in col]
RET_features = [f'RET_{i}' for i in range(1, 21)]


X_train, RET_zscore_features = add_cross_sectional_zscore(X_train, RET_features)
X_test, _ = add_cross_sectional_zscore(X_test, RET_features) 
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
features = features + RET_zscore_features
features = features 
minimal_features = features + ['CROSS_SECTIONAL_AVG_RET1', 'ALLOC_bayesian_ENCODING', 'RET_1_X_ALLOC']



X_train_minimal = X_train.copy()



NUM_BOOST_ROUND = 500

train_dates = X_train_minimal['TS'].unique()
n_splits = 16

scores_minimal = []

max_mean_accuracy = 0
best_ff = None
best_bf = None
best_bfreq = None
smallest_std = float('inf')

oof = X_train_minimal[['TS', 'ALLOCATION']].copy()
oof['actual'] = y_train_bin.values
oof['pred_prob'] = np.nan
oof['fold'] = np.nan
lgbm_params = {
    "objective": "binary",
    "metric": "binary_error",
    "verbosity": -1,
    "learning_rate": 3e-2,
    "max_depth": -1,
    "num_leaves": 31,
    "feature_fraction": 0.3,
    "bagging_fraction": 0.5,
    "bagging_freq": 1,
    "seed": 42,

}
# run your existing CV loop with these params, print ff, bf, mean, std
scores_minimal = []
seed_mean_accuracy = {}
seed_std_accuracy = {}
for seed in [1, 2, 3, 4, 5]:
    date_splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold_idx, (train_date_idx, val_date_idx) in enumerate(
        date_splitter.split(train_dates), 1
    ):
        train_dates_fold = train_dates[train_date_idx]
        val_dates_fold = train_dates[val_date_idx]
        train_row_idx = X_train_minimal.index[
            X_train_minimal['TS'].isin(train_dates_fold)
        ]
        val_row_idx = X_train_minimal.index[
            X_train_minimal['TS'].isin(val_dates_fold)
        ]
        train_df = X_train_minimal.loc[train_row_idx].copy()
        val_df = X_train_minimal.loc[val_row_idx].copy()

        # compute encoding only from the training rows of this fold
        fold_train_targets = y_train_bin.loc[train_row_idx]
        fold_train_allocs = train_df['ALLOCATION']

        alloc_bayesian_encoding, _, _, _ = compute_bayesian_alloc_rate(
            fold_train_targets,
            fold_train_allocs,
            prior_strength=51
        )

        global_rate = fold_train_targets.mean()

        train_df['ALLOC_bayesian_ENCODING'] = (
            train_df['ALLOCATION'].map(alloc_bayesian_encoding).fillna(global_rate)
        )
        val_df['ALLOC_bayesian_ENCODING'] = (
            val_df['ALLOCATION'].map(alloc_bayesian_encoding).fillna(global_rate)
        )

        train_df['RET_1_X_ALLOC'] = train_df['RET_1'] * train_df['ALLOC_bayesian_ENCODING']
        val_df['RET_1_X_ALLOC'] = val_df['RET_1'] * val_df['ALLOC_bayesian_ENCODING']

        X_tr = train_df[minimal_features]
        y_tr = y_train_bin.loc[train_row_idx].values

        X_val = val_df[minimal_features]
        y_val = y_train_bin.loc[val_row_idx].values

        model = lgbm.train(
            lgbm_params,
            lgbm.Dataset(X_tr, label=y_tr),
            num_boost_round=NUM_BOOST_ROUND,
        )
        pred = model.predict(X_val)
        pred_bin = (pred >= 0.5).astype(int)

        score = accuracy_score(y_val, pred_bin)
        scores_minimal.append(score)
        print(f"Fold {fold_idx}: {score:.4f}")

        oof.loc[val_df.index, 'pred_prob'] = pred
        oof.loc[val_df.index, 'fold'] = fold_idx

    mean = np.mean(scores_minimal) * 100
    std = np.std(scores_minimal) * 100
    if mean > max_mean_accuracy:
        max_mean_accuracy = mean
        smallest_std = std
    seed_mean_accuracy[seed] = mean
    seed_std_accuracy[seed] = std
    print(f"\nCombined model accuracy: {mean:.2f}% (+- {std:.2f})")
print("\nSeed-wise mean accuracy:", seed_mean_accuracy)
print("\nSeed-wise std accuracy:", seed_std_accuracy)
no_seed_mean_accuracy = seed_mean_accuracy
no_seed_std_accuracy = seed_std_accuracy
no_beta_average_mean_accuracy = np.mean(list(seed_mean_accuracy.values()))
no_beta_average_std_accuracy = np.mean(list(seed_std_accuracy.values()))
# run_best


full_alloc_encoding, _, _, full_global_rate = compute_bayesian_alloc_rate(
    y_train_bin,
    X_train['ALLOCATION'],
    prior_strength=51
)
X_test['ALLOC_bayesian_ENCODING'] = X_test['ALLOCATION'].map(full_alloc_encoding).fillna(full_global_rate)
X_test['RET_1_X_ALLOC'] = X_test['RET_1'] * X_test['ALLOC_bayesian_ENCODING']
X_test['CROSS_SECTIONAL_AVG_RET1'] = X_test.groupby('TS')['RET_1'].transform('mean')
X_train_minimal['ALLOC_bayesian_ENCODING'] = (
    X_train_minimal['ALLOCATION']
    .map(full_alloc_encoding)
    .fillna(full_global_rate)
)
X_train_minimal['RET_1_X_ALLOC'] = (
    X_train_minimal['RET_1'] * X_train_minimal['ALLOC_bayesian_ENCODING']
)


final_train_data = lgbm.Dataset(X_train_minimal[minimal_features], label=y_train_bin.values)
final_model = lgbm.train(lgbm_params, final_train_data, num_boost_round=NUM_BOOST_ROUND)

preds = final_model.predict(X_test[minimal_features])
preds = pd.DataFrame((preds > 0.501).astype(int), index=sample_submission.index, columns=['target'])
# looking at spread of predictions
train_preds = final_model.predict(X_train_minimal[minimal_features])
train_preds_bin = (train_preds > 0.501).astype(int)
train_accuracy = accuracy_score(y_train_bin.values, train_preds_bin)
print(f"Final model training accuracy: {train_accuracy * 100:.2f}%")
train_preds_df = pd.DataFrame({
    'actual': y_train_bin.values,
    'pred_prob': train_preds,
    'pred_bin': train_preds_bin
}, index=X_train_minimal.index)
print(train_preds_df.head())
train_preds_df.to_csv('data/final_model_train_predictions.csv', index=True)

preds.to_csv('outputs/predictions_lgbm_allocations_cross_more_volume_bin_plus_bays_baseline_bagging_optimisation_changed_folds.csv', index=True)
baseline_features = minimal_features.copy()

beta_features = [f"BETA_{i}" for i in range(2, 21)]
beta_features = [col for col in beta_features if col in X_train_minimal.columns]
def add_beta_and_corr_features(X_train, X_test, RET_features):
    # --- BETA_k: sensitivity to the TS-level market mean ---
    for k in range(2, 21):
        cols = RET_features[:k]
        feat_name = f'BETA_{k}'

        for X in [X_train, X_test]:
            market = X.groupby('TS')[cols].transform('mean')
            R = X[cols].to_numpy(dtype=float)
            M = market[cols].to_numpy(dtype=float)

            Rm = R - R.mean(axis=1, keepdims=True)
            Mm = M - M.mean(axis=1, keepdims=True)
            cov = (Rm * Mm).mean(axis=1)
            var_m = (Mm ** 2).mean(axis=1)
            X[feat_name] = np.where(var_m == 0, 0.0, cov / var_m)

    # --- AVG_CORR_UNIV_k: average pairwise correlation with rest of universe that day ---
    for k in range(2, 21):
        cols = RET_features[:k]
        feat_name = f'AVG_CORR_UNIV_{k}'

        for X in [X_train, X_test]:
            X[feat_name] = 0.0
            for ts, idx in X.groupby('TS').groups.items():
                sub = X.loc[idx, cols].to_numpy(dtype=float)
                n_rows = sub.shape[0]
                if n_rows <= 1:
                    continue
                corr = np.corrcoef(sub)
                corr = np.nan_to_num(corr, nan=0.0)
                np.fill_diagonal(corr, 0.0)
                avg = corr.sum(axis=1) / (n_rows - 1)
                X.loc[idx, feat_name] = avg

    # --- ALLOCATIONS_SORTINO_3: downside-risk-adjusted 3-day mean, averaged across the day's universe ---
    cols = RET_features[:3]
    for X in [X_train, X_test]:
        R = X[cols].to_numpy(dtype=float)
        mean = R.mean(axis=1)
        downside = np.minimum(0.0, R)
        downside_std = np.sqrt((downside ** 2).mean(axis=1))
        X['SORTINO_3'] = mean / (downside_std + 1e-9)
        X['ALLOCATIONS_SORTINO_3'] = X.groupby('TS')['SORTINO_3'].transform('mean')

    # --- ALLOCATIONS_TREND_STRENGTH_10: (max-min)/std over 10 days, averaged across the day's universe ---
    cols = RET_features[:10]
    for X in [X_train, X_test]:
        window = X[cols]
        rng = window.max(axis=1) - window.min(axis=1)
        std = window.std(axis=1)
        X['TREND_STRENGTH_10'] = rng / (std + 1e-9)
        X['ALLOCATIONS_TREND_STRENGTH_10'] = X.groupby('TS')['TREND_STRENGTH_10'].transform('mean')

    new_cols = [
        f'BETA_{i}' for i in range(2, 21)
    ]
    return X_train, X_test, new_cols
models = {
    "baseline": baseline_features,
    "baseline_plus_beta": baseline_features + beta_features,
}
X_train_minimal, X_test, new_features = add_beta_and_corr_features(X_train_minimal, X_test, RET_features)
seed_mean_accuracy = {}
seed_std_accuracy = {}
for seed in [1, 2, 3, 4, 5]:

    lgbm_params = {
        "objective": "binary",
        "metric": "binary_error",
        "verbosity": -1,
        "learning_rate": 3e-2,
        "max_depth": -1,
        "num_leaves": 31,
        "feature_fraction": 0.3,
        "bagging_fraction": 0.5,
        "bagging_freq": 1,
        "seed": seed,

    }
    # run your existing CV loop with these params, print ff, bf, mean, std
    scores_minimal = []
    date_splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold_idx, (train_date_idx, val_date_idx) in enumerate(
        date_splitter.split(train_dates), 1
    ):
        train_dates_fold = train_dates[train_date_idx]
        val_dates_fold = train_dates[val_date_idx]
        train_row_idx = X_train_minimal.index[
            X_train_minimal['TS'].isin(train_dates_fold)
        ]
        val_row_idx = X_train_minimal.index[
            X_train_minimal['TS'].isin(val_dates_fold)
        ]
        train_df = X_train_minimal.loc[train_row_idx].copy()
        val_df = X_train_minimal.loc[val_row_idx].copy()

        # compute encoding only from the training rows of this fold
        fold_train_targets = y_train_bin.loc[train_row_idx]
        fold_train_allocs = train_df['ALLOCATION']

        alloc_bayesian_encoding, _, _, _ = compute_bayesian_alloc_rate(
            fold_train_targets,
            fold_train_allocs,
            prior_strength=51
        )

        global_rate = fold_train_targets.mean()

        train_df['ALLOC_bayesian_ENCODING'] = (
            train_df['ALLOCATION'].map(alloc_bayesian_encoding).fillna(global_rate)
        )
        val_df['ALLOC_bayesian_ENCODING'] = (
            val_df['ALLOCATION'].map(alloc_bayesian_encoding).fillna(global_rate)
        )

        train_df['RET_1_X_ALLOC'] = train_df['RET_1'] * train_df['ALLOC_bayesian_ENCODING']
        val_df['RET_1_X_ALLOC'] = val_df['RET_1'] * val_df['ALLOC_bayesian_ENCODING']

        X_tr = train_df[models['baseline_plus_beta']]
        y_tr = y_train_bin.loc[train_row_idx].values

        X_val = val_df[models['baseline_plus_beta']]
        y_val = y_train_bin.loc[val_row_idx].values

        model = lgbm.train(
            lgbm_params,
            lgbm.Dataset(X_tr, label=y_tr),
            num_boost_round=NUM_BOOST_ROUND,
        )
        pred = model.predict(X_val)
        pred_bin = (pred >= 0.5).astype(int)

        score = accuracy_score(y_val, pred_bin)
        scores_minimal.append(score)
        print(f"Fold {fold_idx}: {score:.4f}")

        oof.loc[val_df.index, 'pred_prob'] = pred
        oof.loc[val_df.index, 'fold'] = fold_idx

    mean = np.mean(scores_minimal) * 100
    std = np.std(scores_minimal) * 100
    if mean > max_mean_accuracy:
        max_mean_accuracy = mean
        smallest_std = std
    seed_mean_accuracy[seed] = mean
    seed_std_accuracy[seed] = std
    print(f"\nCombined model accuracy: {mean:.2f}% (+- {std:.2f})")
# run_best
print("\nSeed-wise mean accuracy:", seed_mean_accuracy)
print("\nSeed-wise std accuracy:", seed_std_accuracy)
beta_average_mean_accuracy = np.mean(list(seed_mean_accuracy.values()))
beta_average_std_accuracy = np.mean(list(seed_std_accuracy.values()))
print(f"beta_average_mean_accuracy: {beta_average_mean_accuracy:.2f}% (+- {beta_average_std_accuracy:.2f}%)")
print(f"no_beta_average_mean_accuracy: {no_beta_average_mean_accuracy:.2f}% (+- {no_beta_average_std_accuracy:.2f}%)")
paired_differences = np.array(list(seed_mean_accuracy.values())) - np.array(list(no_seed_mean_accuracy.values()))
paired_std_differences = np.array(list(seed_std_accuracy.values())) - np.array(list(no_seed_std_accuracy.values()))

print("Mean improvement:", paired_differences.mean())
print("Beta wins:", (paired_differences > 0).mean())
print("Std improvement:", paired_std_differences.mean())
print("Beta wins (std):", (paired_std_differences > 0).mean())

full_alloc_encoding, _, _, full_global_rate = compute_bayesian_alloc_rate(
    y_train_bin,
    X_train['ALLOCATION'],
    prior_strength=51
)
X_test['ALLOC_bayesian_ENCODING'] = X_test['ALLOCATION'].map(full_alloc_encoding).fillna(full_global_rate)
X_test['RET_1_X_ALLOC'] = X_test['RET_1'] * X_test['ALLOC_bayesian_ENCODING']
X_test['CROSS_SECTIONAL_AVG_RET1'] = X_test.groupby('TS')['RET_1'].transform('mean')
X_train_minimal['ALLOC_bayesian_ENCODING'] = (
    X_train_minimal['ALLOCATION']
    .map(full_alloc_encoding)
    .fillna(full_global_rate)
)
X_train_minimal['RET_1_X_ALLOC'] = (
    X_train_minimal['RET_1'] * X_train_minimal['ALLOC_bayesian_ENCODING']
)



final_train_data = lgbm.Dataset(X_train_minimal[minimal_features], label=y_train_bin.values)
final_model = lgbm.train(lgbm_params, final_train_data, num_boost_round=NUM_BOOST_ROUND)

preds = final_model.predict(X_test[minimal_features])
preds = pd.DataFrame((preds > 0.501).astype(int), index=sample_submission.index, columns=['target'])
# looking at spread of predictions
train_preds = final_model.predict(X_train_minimal[minimal_features])
train_preds_bin = (train_preds > 0.501).astype(int)
train_accuracy = accuracy_score(y_train_bin.values, train_preds_bin)
print(f"Final model training accuracy: {train_accuracy * 100:.2f}%")
train_preds_df = pd.DataFrame({
    'actual': y_train_bin.values,
    'pred_prob': train_preds,
    'pred_bin': train_preds_bin
}, index=X_train_minimal.index)
print(train_preds_df.head())
train_preds_df.to_csv('data/final_model_train_predictions.csv', index=True)

preds.to_csv('outputs/predictions_lgbm_allocations_cross_more_volume_bin_plus_bays_baseline_bagging_optimisation_changed_folds_beta.csv', index=True)