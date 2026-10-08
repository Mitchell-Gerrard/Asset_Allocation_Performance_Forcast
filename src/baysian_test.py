import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgbm
import scipy
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.calibration import calibration_curve
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

RET_features = [f'RET_{i}' for i in range(1, 21)]
x_train, RET_zscore_features = add_cross_sectional_zscore(X_train, RET_features)
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
minimal_features = features + ['CROSS_SECTIONAL_AVG_RET1', 'ALLOC_bayesian_ENCODING']
X_train_minimal = X_train.copy()

lgbm_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_threads": 50,
    "verbosity": -1,
    'learning_rate': 1e-2,
    'max_depth': 3,
}
NUM_BOOST_ROUND = 500

train_dates = X_train_minimal['TS'].unique()
n_splits = 8
scores_minimal = []

splits = KFold(n_splits=n_splits, random_state=0, shuffle=True).split(train_dates)
oof = X_train_minimal[['TS', 'ALLOCATION']].copy()
oof['actual'] = y_train_bin.values
oof['pred_prob'] = np.nan
oof['fold'] = np.nan
for i, (local_train_dates_ids, local_test_dates_ids) in enumerate(splits):
    local_train_dates = train_dates[local_train_dates_ids]
    local_test_dates = train_dates[local_test_dates_ids]

    local_train_ids = X_train_minimal['TS'].isin(local_train_dates)
    local_test_ids = X_train_minimal['TS'].isin(local_test_dates)

    # --- build the allocation target encoding using ONLY this fold's training rows ---
    fold_train_targets = y_train_bin.loc[local_train_ids]
    fold_train_allocs = X_train_minimal.loc[local_train_ids, 'ALLOCATION']

    alloc_encoding = (
        pd.DataFrame({'ALLOCATION': fold_train_allocs, 'target': fold_train_targets.values})
        .groupby('ALLOCATION')['target']
        .mean()
    )
    global_rate = fold_train_targets.mean()

    # Use the Bayesian approach to compute the allocation encoding
    alloc_bayesian_encoding, _, _, _ = compute_bayesian_alloc_rate(
        fold_train_targets, fold_train_allocs, prior_strength=51
    )

    X_train_minimal.loc[local_train_ids, 'ALLOC_bayesian_ENCODING'] = (
        X_train_minimal.loc[local_train_ids, 'ALLOCATION'].map(alloc_bayesian_encoding).fillna(global_rate)
    )
    X_train_minimal.loc[local_test_ids, 'ALLOC_bayesian_ENCODING'] = (
        X_train_minimal.loc[local_test_ids, 'ALLOCATION'].map(alloc_bayesian_encoding).fillna(global_rate)
    )
    # --- end encoding block ---

    X_local_train = X_train_minimal.loc[local_train_ids, minimal_features]
    y_local_train = y_train_bin.loc[local_train_ids]

    X_local_test = X_train_minimal.loc[local_test_ids, minimal_features]
    y_local_test = y_train_bin.loc[local_test_ids]

    train_data = lgbm.Dataset(X_local_train, label=y_local_train.values)

    model = lgbm.train(lgbm_params, train_data, num_boost_round=NUM_BOOST_ROUND)
    y_local_pred = model.predict(X_local_test, num_threads=lgbm_params['num_threads'])

    y_local_pred_bin = (y_local_pred > 0.501).astype(int)
    score = accuracy_score(y_local_test.values, y_local_pred_bin)
    scores_minimal.append(score)
    print(f"Fold {i+1} - Accuracy: {score * 100:.2f}%")
    oof.loc[local_test_ids, 'pred_prob'] = y_local_pred
    oof.loc[local_test_ids, 'fold'] = i

mean = np.mean(scores_minimal) * 100
std = np.std(scores_minimal) * 100
print(f"\nCombined model accuracy: {mean:.2f}% (+- {std:.2f})")
oof['pred'] = (oof['pred_prob'] >= 0.5).astype(int)

print(confusion_matrix(oof['actual'], oof['pred']))

print(classification_report(
    oof['actual'],
    oof['pred'],
    digits=3
))
errors = oof[oof['actual'] != oof['pred']].copy()

print(errors['pred_prob'].describe())
false_positive = oof[
    (oof['actual'] == 0) &
    (oof['pred'] == 1)
]

false_negative = oof[
    (oof['actual'] == 1) &
    (oof['pred'] == 0)
]

print("False positives:")
print(false_positive['pred_prob'].describe())

print("\nFalse negatives:")
print(false_negative['pred_prob'].describe())
oof['confidence'] = np.maximum(
    oof['pred_prob'],
    1 - oof['pred_prob']
)

high_conf_errors = (
    oof[oof['actual'] != oof['pred']]
    .sort_values('confidence', ascending=False)
)

print(high_conf_errors.head(20))
max_score = 0
for threshold in np.arange(0.1, 0.91, 0.001):

    pred = (oof['pred_prob'] >= threshold).astype(int)

    score = accuracy_score(oof['actual'], pred)
    if score > max_score:
        max_score = score
        best_threshold = threshold
    plt.scatter(threshold, score, color='blue')
print(f"Threshold: {threshold:.3f} - Accuracy: {score * 100:.2f}%")
print(f"\nBest threshold: {best_threshold:.3f} - Accuracy: {max_score * 100:.2f}%")
plt.savefig('outputs/accuracy_vs_threshold_lgbm_allocations_baysian_cross_more_volume_bin_plus_baseline.png')
plt.clf()
prob_true, prob_pred = calibration_curve(
    oof['actual'],
    oof['pred_prob'],
    n_bins=10
)

plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("Predicted probability")
plt.ylabel("Actual frequency")
plt.title("Calibration")
plt.savefig('outputs/calibration_curve_lgbm_allocations_baysian_cross_more_volume_bin_plus_baseline.png')
plt.clf()
allocation_stats = (
    pd.DataFrame({
        'ALLOCATION': X_train_minimal['ALLOCATION'],
        'target': y_train_bin
    })
    .groupby('ALLOCATION')
    .agg(
        target_rate=('target', 'mean'),
        count=('target', 'size')
    )
    .sort_values('count', ascending=False)
)

print(allocation_stats)
alloc_counts = (
    X_train_minimal['ALLOCATION']
    .value_counts()
    .rename('allocation_count')
)

oof = oof.join(
    alloc_counts,
    on='ALLOCATION'
)

oof['error'] = (
    oof['actual'] != oof['pred']
).astype(int)
oof.groupby(
    pd.cut(oof['allocation_count'], bins=[0, 5, 20, 100, 500, 5000, np.inf])
)['error'].mean()
explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_local_test)

shap.summary_plot(
    shap_values,
    X_local_test,show=False,max_display=5
)
plt.savefig("outputs/shap_summary.png", dpi=300, bbox_inches="tight")
plt.clf()
shap_values = explainer.shap_values(X_local_test)

# For binary classification, handle both common SHAP formats
if isinstance(shap_values, list):
    sv = shap_values[1]
else:
    sv = shap_values

shap_df = pd.DataFrame(
    sv,
    columns=X_local_test.columns,
    index=X_local_test.index
)

allocation_shap = shap_df['ALLOC_bayesian_ENCODING']

print(allocation_shap.describe())
plt.figure(figsize=(10, 6))
plt.hist(allocation_shap, bins=50)
plt.xlabel("SHAP value for ALLOCATION")
plt.ylabel("Count")
plt.title("ALLOCATION SHAP contributions")
plt.savefig("outputs/allocation_shap_histogram.png", dpi=300, bbox_inches="tight")
plt.clf()
oof['residual'] = (
    oof['actual'] - oof['pred_prob']
)

#for feature in minimal_features:
    #print(feature,oof.groupby(pd.qcut(X_train_minimal.loc[oof.index, feature],q=10,duplicates='drop'))['residual'].mean())
# --- final model for submission: encode using the FULL training set ---
full_alloc_encoding, _, _, full_global_rate = compute_bayesian_alloc_rate(
    y_train_bin,
    X_train['ALLOCATION'],
    prior_strength=51
)
X_test['ALLOC_bayesian_ENCODING'] = X_test['ALLOCATION'].map(full_alloc_encoding).fillna(full_global_rate)
X_test['CROSS_SECTIONAL_AVG_RET1'] = X_test.groupby('TS')['RET_1'].transform('mean')
X_train_minimal['ALLOC_bayesian_ENCODING'] = (
    X_train_minimal['ALLOCATION']
    .map(full_alloc_encoding)
    .fillna(full_global_rate)
)


final_train_data = lgbm.Dataset(X_train_minimal[minimal_features], label=y_train_bin.values)
final_model = lgbm.train(lgbm_params, final_train_data, num_boost_round=NUM_BOOST_ROUND)

preds_lgbm = final_model.predict(X_test[minimal_features], num_threads=lgbm_params['num_threads'])
plt.hist(preds_lgbm, bins=10)
preds_lgbm = pd.DataFrame((preds_lgbm > 0.501).astype(int), index=sample_submission.index, columns=['target'])
# looking at spread of predictions

plt.savefig('outputs/predictions_lgbm_allocations_cross_more_volume_plus_bays_baseline_histogram.png')
plt.clf()
preds_lgbm.to_csv('outputs/predictions_lgbm_allocations_cross_more_volume_bin_plus_bays_baseline.csv', index=True)