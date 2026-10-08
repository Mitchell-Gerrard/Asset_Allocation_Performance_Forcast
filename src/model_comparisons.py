import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgbm
import scipy
from sklearn.model_selection import KFold, TimeSeriesSplit, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.calibration import calibration_curve
import shap
import scipy.stats as stats

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Conv1D, GlobalAveragePooling1D,
    Dense, Dropout, Concatenate
)
def add_cross_sectional_zscore(X, base_features, suffix='_ZSCORE'):
    """
    For each feature in base_features, compute its z-score relative to the
    cross-section of all allocations present at the same TS.
    z = (x - mean(x at this TS)) / std(x at this TS)

    This isolates relative strength on a given day, independent of that
    day's overall market regime (volatile vs calmf days), which raw values
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
def build_transition_feature(X, n_bins=4):
    """Builds the raw (unbinned) current-state array for every row, using its
    own 20-day RET sequence. This part uses NO target info, so it's safe to
    compute once outside the fold loop."""
    ret_cols_ordered = [f'RET_{i}' for i in range(20, 0, -1)]
    ret_seq = X[ret_cols_ordered].to_numpy()
    return ret_seq  # raw sequence; binning happens per-fold using fold-specific edges

def compute_fold_transition_lookup(ret_seq_train, n_bins=4):
    """Fits bin edges AND the transition matrix using only this fold's training rows.
    Returns a function that maps a row's CURRENT (most recent) state to
    P(next day = highest bin | current state) -- the 'P(next positive-extreme)' feature."""
    clean_values = ret_seq_train[~np.isnan(ret_seq_train)]
    edges = np.quantile(clean_values, np.linspace(0, 1, n_bins + 1))

    def get_states(ret_seq):
        ret_seq_clean = np.nan_to_num(ret_seq, nan=0.0)
        return np.digitize(ret_seq_clean, edges[1:-1])

    states_train = get_states(ret_seq_train)

    transition_counts = np.zeros((n_bins, n_bins))
    for row in states_train:
        for t in range(len(row) - 1):
            transition_counts[row[t], row[t + 1]] += 1
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    transition_matrix = transition_counts / np.where(row_sums == 0, 1, row_sums)

    # "current state" = state of the most recent day (RET_1, last column after our ordering)
    def lookup_feature(ret_seq_query):
        query_states = get_states(ret_seq_query)
        current_state = query_states[:, -1]  # last column = most recent day (RET_1)
        # P(next day's state == highest bin | current state)
        return transition_matrix[current_state, n_bins - 1]

    return lookup_feature
X_train = pd.read_csv('data/X_train.csv', index_col='ROW_ID')
X_test = pd.read_csv('data/X_test.csv', index_col='ROW_ID')
y_train = pd.read_csv('data/y_train.csv', index_col='ROW_ID')
y_train_bin = (y_train['target'] > 0).astype(int)
sample_submission = pd.read_csv('data/sample_submission.csv', index_col='ROW_ID')


RET_features = [f'RET_{i}' for i in range(1, 21)]

SIGNED_VOLUME_features = [f'SIGNED_VOLUME_{i}' for i in range(1, 21)]
TURNOVER_features = ['MEDIAN_DAILY_TURNOVER']

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
minimal_features = features 
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
minimal_features = features 





minimal_features += ["ALLOCATION","GROUP"]
#X_train["ALLOCATION"] = np.random.permutation(X_train["ALLOCATION"].values)
X_train["ALLOCATION"] = X_train["ALLOCATION"].astype("category")
X_test["ALLOCATION"] = X_test["ALLOCATION"].astype("category")
X_test["GROUP"] = X_test["GROUP"].astype("category")
X_train["GROUP"] = X_train["GROUP"].astype("category")

print("Minimal features:", minimal_features)
X_train_minimal = X_train.copy()
models_acuracy = {}
#lgbm_tree_model
print("Training LightGBM model with minimal features...")
lgbm_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_threads": 50,
    "verbosity": -1,
    'learning_rate': 1e-2,
    'max_depth': 3,
    'feature_fraction': 0.3, 'bagging_fraction': 0.5, 'bagging_freq': 1,
}
NUM_BOOST_ROUND = 500

train_dates = np.sort(X_train_minimal['TS'].unique())
n_splits = 8
scores_minimal = []

splits = TimeSeriesSplit(n_splits=n_splits).split(train_dates)
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
    ret_cols_ordered = [f'RET_{i}' for i in range(20, 0, -1)]

    ret_seq_train_fold = X_train_minimal.loc[local_train_ids, ret_cols_ordered].to_numpy()
    lookup_fn = compute_fold_transition_lookup(ret_seq_train_fold, n_bins=4)

    ret_seq_train_query = X_train_minimal.loc[local_train_ids, ret_cols_ordered].to_numpy()
    ret_seq_test_query = X_train_minimal.loc[local_test_ids, ret_cols_ordered].to_numpy()

    X_train_minimal.loc[local_train_ids, 'MARKOV_P_UP'] = lookup_fn(ret_seq_train_query)
    X_train_minimal.loc[local_test_ids, 'MARKOV_P_UP'] = lookup_fn(ret_seq_test_query)


    X_train_minimal.loc[local_train_ids, 'ALLOC_bayesian_ENCODING'] = (
        X_train_minimal.loc[local_train_ids, 'ALLOCATION'].map(alloc_bayesian_encoding).fillna(global_rate)
    )
    X_train_minimal.loc[local_test_ids, 'ALLOC_bayesian_ENCODING'] = (
        X_train_minimal.loc[local_test_ids, 'ALLOCATION'].map(alloc_bayesian_encoding).fillna(global_rate)
    )
    X_train_minimal['RET_1_X_ALLOC'] = (
    X_train_minimal['RET_1'] * X_train_minimal['ALLOC_bayesian_ENCODING']
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

#for feature in minimal_features:
    #print(feature,oof.groupby(pd.qcut(X_train_minimal.loc[oof.index, feature],q=10,duplicates='drop'))['residual'].mean())
# --- final model for submission: encode using the FULL training set ---
full_alloc_encoding, _, _, full_global_rate = compute_bayesian_alloc_rate(
    y_train_bin,
    X_train['ALLOCATION'],
    prior_strength=51
)
X_test['CROSS_SECTIONAL_AVG_RET1'] = X_test.groupby('TS')['RET_1'].transform('mean')



date_split_index = int(len(train_dates) * 0.8)
training_dates = train_dates[:date_split_index]
validation_dates = train_dates[date_split_index:]
lgbm_train_mask = X_train_minimal['TS'].isin(training_dates)
lgbm_val_mask = X_train_minimal['TS'].isin(validation_dates)
x_tr = X_train_minimal.loc[lgbm_train_mask, minimal_features]
x_val = X_train_minimal.loc[lgbm_val_mask, minimal_features]
y_tr = y_train_bin.loc[lgbm_train_mask]
y_val = y_train_bin.loc[lgbm_val_mask]
print(
    f"LightGBM date validation: train through {training_dates[-1]}, "
    f"validate from {validation_dates[0]}"
)
final_train_data = lgbm.Dataset(x_tr, label=y_tr.values)
final_model = lgbm.train(lgbm_params, final_train_data, num_boost_round=NUM_BOOST_ROUND)

preds_lgbm = final_model.predict(X_test[minimal_features], num_threads=lgbm_params['num_threads'])
plt.hist(preds_lgbm, bins=10)
preds_lgbm = pd.DataFrame((preds_lgbm > 0.501).astype(int), index=sample_submission.index, columns=['target'])
# looking at spread of predictions

train_preds = final_model.predict(x_val, num_threads=lgbm_params['num_threads'])
train_preds_bin = (train_preds > 0.501).astype(int)
train_accuracy = accuracy_score(y_val.values, train_preds_bin)
print(f"Final model training accuracy: {train_accuracy * 100:.2f}%")

models_acuracy['lgbm_tree_model'] = train_accuracy

# LightGBM handles the pandas categorical column natively.  The other models
# need numeric columns, so one-hot encode allocation while keeping train/test
# categories aligned.
numeric_model_inputs = pd.concat(
    [X_train_minimal[minimal_features], X_test[minimal_features]],
    axis=0,
)
numeric_model_inputs = pd.get_dummies(
    numeric_model_inputs,
    columns=['ALLOCATION','GROUP'],
    dtype=float,
)
X_train_numeric = numeric_model_inputs.iloc[:len(X_train_minimal)].copy()
X_test_numeric = numeric_model_inputs.iloc[len(X_train_minimal):].copy()

date_split_index = int(len(train_dates) * 0.8)
validation_dates = train_dates[date_split_index:]
training_dates = train_dates[:date_split_index]
train_row_ids = np.flatnonzero(X_train_minimal['TS'].isin(training_dates).to_numpy())
val_row_ids = np.flatnonzero(X_train_minimal['TS'].isin(validation_dates).to_numpy())
print(
    f"Date validation: train through {training_dates[-1]}, "
    f"validate from {validation_dates[0]} ({len(train_row_ids):,}/{len(val_row_ids):,} rows)"
)

#logistic regression model
print("Training Logistic Regression model with minimal features...")
X_tr = X_train_numeric.iloc[train_row_ids]
X_val = X_train_numeric.iloc[val_row_ids]

log_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        C=0.1,
        max_iter=1000,
    ))
])

log_model.fit(X_tr, y_train_bin.iloc[train_row_ids])

train_pred = log_model.predict(X_tr)
val_pred = log_model.predict(X_val)
train_accuracy = accuracy_score(y_train_bin.iloc[train_row_ids], train_pred)
val_accuracy = accuracy_score(y_train_bin.iloc[val_row_ids], val_pred)
print("Logistic Regression train accuracy:", train_accuracy)
print("Logistic Regression validation accuracy:", val_accuracy)
models_acuracy['logistic_regression'] = val_accuracy
#extra trees model
print("Training Extra Trees model with minimal features...")
X_tr = X_train_numeric
X_te = X_test_numeric
"""
for n in [10, 25, 50, 100, 250, 500, 1000]:
    model = ExtraTreesClassifier(
        n_estimators=n,
        max_features=0.2,
        min_samples_leaf=50,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_tr.iloc[train_row_ids], y_train_bin.iloc[train_row_ids])

    print(
        f"{n:4d} trees | "
        f"train: {model.score(X_tr.iloc[train_row_ids], y_train_bin.iloc[train_row_ids]):.4f} | "
        f"val: {model.score(X_tr.iloc[val_row_ids], y_train_bin.iloc[val_row_ids]):.4f}"
    )
    """
 
extra_model = ExtraTreesClassifier(
    n_estimators=100,
    max_features=0.2,
    min_samples_leaf=50,
    n_jobs=-1,
    random_state=42,
    verbose=1
)
extra_model.fit(X_tr.iloc[train_row_ids], y_train_bin.iloc[train_row_ids])

train_pred = extra_model.predict(X_tr.iloc[train_row_ids])
val_pred = extra_model.predict(X_tr.iloc[val_row_ids])
for name, importance in sorted(
    zip(X_train_numeric.columns, extra_model.feature_importances_),
    key=lambda x: x[1],
    reverse=True
)[:20]:
    print(name, importance)
test_pred = extra_model.predict(X_te)
preds_df=pd.DataFrame(test_pred, index=sample_submission.index, columns=['target'])
preds_df.to_csv('outputs/extra_trees_predictions.csv')
train_accuracy = accuracy_score(y_train_bin.iloc[train_row_ids], train_pred)
val_accuracy = accuracy_score(y_train_bin.iloc[val_row_ids], val_pred)
print("Extra Trees train accuracy:", train_accuracy)
print("Extra Trees validation accuracy:", val_accuracy)
models_acuracy['extra_trees'] = val_accuracy
#xgboost model
print("Training XGBoost model with minimal features...")
X_tr = X_train_numeric
X_te = X_test_numeric

xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.5,
    min_child_weight=50,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=-1,
    random_state=42
)

xgb_model.fit(
    X_tr.iloc[train_row_ids],
    y_train_bin.iloc[train_row_ids],
    eval_set=[(X_tr.iloc[val_row_ids], y_train_bin.iloc[val_row_ids])],
    verbose=False,
)

for name, importance in sorted(
    zip(X_train_numeric.columns, xgb_model.feature_importances_),
    key=lambda x: x[1],
    reverse=True
)[:20]:
    print(name, importance)

train_pred = xgb_model.predict(X_tr.iloc[train_row_ids])
val_pred = xgb_model.predict(X_tr.iloc[val_row_ids])
train_accuracy = accuracy_score(y_train_bin.iloc[train_row_ids], train_pred)
val_accuracy = accuracy_score(y_train_bin.iloc[val_row_ids], val_pred)
print("XGBoost train accuracy:", train_accuracy)
print("XGBoost validation accuracy:", val_accuracy)
models_acuracy['xgboost'] = val_accuracy

# Branch-specific XGBoost models: let each model learn exceptions to RET_1.
print("Training branch-specific XGBoost models...")
branch_models = {}
branch_validation_pred = np.empty(len(val_row_ids), dtype=int)
ret_sign = np.sign(X_train_minimal['RET_1'].to_numpy()).astype(int)

for branch_sign in (-1, 1):
    branch_train_ids = train_row_ids[ret_sign[train_row_ids] == branch_sign]
    branch_val_positions = np.flatnonzero(
        ret_sign[val_row_ids] == branch_sign
    )

    branch_model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.5,
        min_child_weight=50,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=100 + branch_sign,
    )
    branch_model.fit(
        X_tr.iloc[branch_train_ids],
        y_train_bin.iloc[branch_train_ids],
        eval_set=[
            (
                X_tr.iloc[val_row_ids[ret_sign[val_row_ids] == branch_sign]],
                y_train_bin.iloc[val_row_ids[ret_sign[val_row_ids] == branch_sign]],
            )
        ],
        verbose=False,
    )
    branch_models[branch_sign] = branch_model
    branch_validation_pred[branch_val_positions] = branch_model.predict(
        X_tr.iloc[val_row_ids[branch_val_positions]]
    ).astype(int)

branch_accuracy = accuracy_score(
    y_train_bin.iloc[val_row_ids], branch_validation_pred
)
print("Branch-specific XGBoost validation accuracy:", branch_accuracy)
models_acuracy['xgboost_branch'] = branch_accuracy
#cat bost 
print("Training CatBoost model with minimal features...")
X_tr = X_train_minimal[minimal_features].copy()
X_te = X_test[minimal_features].copy()
X_tr['ALLOCATION'] = X_tr['ALLOCATION'].astype(str)
X_te['ALLOCATION'] = X_te['ALLOCATION'].astype(str)

cat_model = CatBoostClassifier(
    iterations=500,
    depth=7,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="Accuracy",
    verbose=False,
    random_seed=42
)

cat_model.fit(
    X_tr.iloc[train_row_ids],
    y_train_bin.iloc[train_row_ids],
    cat_features=['ALLOCATION','GROUP'],
    eval_set=(X_tr.iloc[val_row_ids], y_train_bin.iloc[val_row_ids]),
)

train_pred = cat_model.predict(X_tr.iloc[train_row_ids]).ravel()
val_pred = cat_model.predict(X_tr.iloc[val_row_ids]).ravel()
train_accuracy = accuracy_score(y_train_bin.iloc[train_row_ids], train_pred)
val_accuracy = accuracy_score(y_train_bin.iloc[val_row_ids], val_pred)
print("CatBoost train accuracy:", train_accuracy)
print("CatBoost validation accuracy:", val_accuracy)
models_acuracy['catboost'] = val_accuracy
test_pred = cat_model.predict(X_te).ravel()
preds_df=pd.DataFrame(test_pred, index=sample_submission.index, columns=['target'])
preds_df.to_csv('outputs/catboost_predictions.csv', index=True)
#convolutional model
print("Training Convolutional Neural Network model with minimal features...")
# --------------------------------------------------
# 1. Tabular features = ALL your minimal_features
# --------------------------------------------------

X_tr_tab = X_train_numeric.copy()
X_te_tab = X_test_numeric.copy()

imputer = SimpleImputer(strategy="median")

X_tr_tab = imputer.fit_transform(X_tr_tab)
X_te_tab = imputer.transform(X_te_tab)

scaler = StandardScaler()

X_tr_tab = scaler.fit_transform(X_tr_tab)
X_te_tab = scaler.transform(X_te_tab)


# --------------------------------------------------
# 2. Sequential features
#    20 lags × 2 channels
# --------------------------------------------------

ret_cols = [f"RET_{i}" for i in range(20, 0, -1)]
vol_cols = [f"SIGNED_VOLUME_{i}" for i in range(20, 0, -1)]

ret_train = X_train_minimal[ret_cols].fillna(0).values
vol_train = X_train_minimal[vol_cols].fillna(0).values

ret_test = X_test[ret_cols].fillna(0).values
vol_test = X_test[vol_cols].fillna(0).values

X_tr_seq = np.stack(
    [ret_train, vol_train],
    axis=-1
)

X_te_seq = np.stack(
    [ret_test, vol_test],
    axis=-1
)


# scale sequential channels using TRAIN only
for c in range(2):
    mean = X_tr_seq[:, :, c].mean()
    std = X_tr_seq[:, :, c].std()

    X_tr_seq[:, :, c] = (
        X_tr_seq[:, :, c] - mean
    ) / (std + 1e-8)

    X_te_seq[:, :, c] = (
        X_te_seq[:, :, c] - mean
    ) / (std + 1e-8)


# --------------------------------------------------
# 3. CNN branch
# --------------------------------------------------

seq_input = Input(shape=(20, 2))

x = Conv1D(
    32,
    kernel_size=3,
    activation="relu"
)(seq_input)

x = Conv1D(
    32,
    kernel_size=3,
    activation="relu"
)(x)

x = GlobalAveragePooling1D()(x)

x = Dense(32, activation="relu")(x)


# --------------------------------------------------
# 4. Tabular branch
# --------------------------------------------------

tab_input = Input(shape=(X_tr_tab.shape[1],))

t = Dense(64, activation="relu")(tab_input)

t = Dropout(0.2)(t)

t = Dense(32, activation="relu")(t)


# --------------------------------------------------
# 5. Combine both
# --------------------------------------------------

combined = Concatenate()([x, t])

combined = Dense(
    64,
    activation="relu"
)(combined)

combined = Dropout(0.2)(combined)

output = Dense(
    1,
    activation="sigmoid"
)(combined)


cnn_model = Model(
    inputs=[seq_input, tab_input],
    outputs=output
)


cnn_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# --------------------------------------------------
# 6. Train
# --------------------------------------------------

cnn_model.fit(
    [X_tr_seq[train_row_ids], X_tr_tab[train_row_ids]],
    y_train_bin.iloc[train_row_ids].values,
    validation_data=(
        [X_tr_seq[val_row_ids], X_tr_tab[val_row_ids]],
        y_train_bin.iloc[val_row_ids].values,
    ),
    epochs=10,
    batch_size=2048,
    verbose=1
)


train_pred_prob = cnn_model.predict(
    [X_tr_seq[train_row_ids], X_tr_tab[train_row_ids]],
    batch_size=4096,
    verbose=0
).ravel()
val_pred_prob = cnn_model.predict(
    [X_tr_seq[val_row_ids], X_tr_tab[val_row_ids]],
    batch_size=4096,
    verbose=0
).ravel()

train_pred = (train_pred_prob >= 0.5).astype(int)
val_pred = (val_pred_prob >= 0.5).astype(int)
train_accuracy = accuracy_score(y_train_bin.iloc[train_row_ids], train_pred)
val_accuracy = accuracy_score(y_train_bin.iloc[val_row_ids], val_pred)

print(
    "CNN train accuracy:", train_accuracy
)
print("CNN validation accuracy:", val_accuracy)
models_acuracy['cnn_model'] = val_accuracy
print("Model accuracies:", models_acuracy)

print("dates:", X_train['TS'].nunique(),
      "| allocs:", X_train['ALLOCATION'].nunique(),
      "| groups:", X_train['GROUP'].nunique())
print(X_train.groupby('TS').size().describe())
print("max groups per allocation:", X_train.groupby('ALLOCATION')['GROUP'].nunique().max())

# is the target market-driven or cross-sectionally balanced?
day_up = y_train_bin.groupby(X_train['TS']).mean()
print(day_up.describe())
print("std if rows were independent:", 0.5 / np.sqrt(X_train.groupby('TS').size().mean()))

# are TS labels chronological? (consecutive labels should shift each window by one day)
ts_sorted = np.sort(X_train['TS'].unique())
nxt = dict(zip(ts_sorted[:-1], ts_sorted[1:]))
a = X_train[['TS', 'ALLOCATION', 'RET_1']].copy()
a['TS_next'] = a['TS'].map(nxt)
b = X_train[['TS', 'ALLOCATION', 'RET_2']]
m = a.merge(b, left_on=['TS_next', 'ALLOCATION'], right_on=['TS', 'ALLOCATION'])
print("consecutive labels with matching windows:", np.isclose(m['RET_1'], m['RET_2']).mean())
y = y_train_bin.to_numpy()
ts = X_train['TS'].to_numpy()

def per_day_acc(ts, y, p):
    s = pd.Series(y == p).groupby(ts).mean()
    return s.mean(), s.std() / np.sqrt(len(s))

rules = {
    'always up': np.ones(len(y), dtype=int),
    'own RET_1 < 0': (X_train['RET_1'] < 0).astype(int).to_numpy(),
    'own RET_1 > 0': (X_train['RET_1'] > 0).astype(int).to_numpy(),
    'day mean RET_1 < 0': (X_train.groupby('TS')['RET_1'].transform('mean') < 0).astype(int).to_numpy(),
    'day+group mean RET_1 < 0': (X_train.groupby(['TS', 'GROUP'])['RET_1'].transform('mean') < 0).astype(int).to_numpy(),
    'day+group mean 5d < 0': (X_train.groupby(['TS', 'GROUP'])['AVERAGE_PERF_5'].transform('mean') < 0).astype(int).to_numpy(),
    'day+group mean 5d > 0': (X_train.groupby(['TS', 'GROUP'])['AVERAGE_PERF_5'].transform('mean') > 0).astype(int).to_numpy(),
}
for name, p in rules.items():
    acc, se = per_day_acc(ts, y, p)
    print(f"{name:28s} {acc:.4f} +- {se:.4f}")
# 1. every lag, on its own: sign(RET_k) and sign(SIGNED_VOLUME_k) as predictors
rows = []
for k in range(1, 21):
    for col in (f'RET_{k}', f'SIGNED_VOLUME_{k}'):
        acc, se = per_day_acc(ts, y, (X_train[col] > 0).astype(int).to_numpy())
        rows.append((col, acc, se))
lag_tab = pd.DataFrame(rows, columns=['feature', 'acc', 'se'])
lag_tab['dist_from_50'] = (lag_tab['acc'] - 0.5).abs()
print(lag_tab.sort_values('dist_from_50', ascending=False).head(12))

# 2. where does the RET_1 momentum live?
p = (X_train['RET_1'] > 0).astype(int).to_numpy()
d = X_train[['ALLOCATION', 'GROUP']].copy()
d['hit'] = (p == y)

print(d.groupby('GROUP')['hit'].agg(['mean', 'size']))

alloc_hit = d.groupby('ALLOCATION')['hit'].mean()
print(alloc_hit.describe())   # spread across allocations vs ~0.012 expected from pure noise

shock = X_train['RET_1'].abs() / (X_train[RET_features].std(axis=1) + 1e-12)
d['shock_bin'] = pd.qcut(shock, 5, duplicates='drop')
print(d.groupby('shock_bin', observed=True)['hit'].mean())

d['vol_agree'] = (np.sign(X_train['SIGNED_VOLUME_1']) == np.sign(X_train['RET_1'])).astype(int)
print(d.groupby('vol_agree')['hit'].agg(['mean', 'size']))

# 3. does a model beat the rule? paired per-day difference with a standard error
def paired_vs_rule(ts, y, p_model, p_rule):
    diff = pd.Series((y == p_model).astype(float) - (y == p_rule).astype(float)).groupby(ts).mean()
    return diff.mean(), diff.std() / np.sqrt(len(diff))

m = oof['pred_prob'].notna().to_numpy()
p_model = (oof['pred_prob'][m] > 0.501).astype(int).to_numpy()
print("LGBM OOF minus RET_1>0 rule:", paired_vs_rule(ts[m], y[m], p_model, p[m]))
#next block
print("next block")
r = X_train[RET_features].to_numpy(dtype=float)       # column 0 = RET_1
ret1 = r[:, 0]
std_ex = np.nanstd(r[:, 1:], axis=1)                  # vol excluding RET_1 itself
alloc_vol = (pd.Series(np.nanstd(r, axis=1), index=X_train.index)
             .groupby(X_train['ALLOCATION']).transform('mean').to_numpy())

variants = {
    'RET_1 / std20 (current)':     ret1 / (np.nanstd(r, axis=1) + 1e-12),
    'RET_1 / std(2..20)':          ret1 / (std_ex + 1e-12),
    'RET_1 / alloc long-run vol':  ret1 / (alloc_vol + 1e-12),
}

# 1. P(up) by signed decile -- is it symmetric? where does it become informative?
for name, z in variants.items():
    bins = pd.qcut(z, 10, duplicates='drop')
    print(name)
    print(pd.Series(y).groupby(bins, observed=True).mean().round(4).to_string(), "\n")

# 2. gated rule: follow the sign of RET_1 only above a |z| threshold, else predict up
def gated_rule(z, ret1, q):
    thr = np.nanquantile(np.abs(z), q)
    return np.where(np.abs(z) >= thr, (ret1 > 0).astype(int), 1)

for name, z in variants.items():
    for q in [0.0, 0.2, 0.4, 0.6]:
        acc, se = per_day_acc(ts, y, gated_rule(z, ret1, q))
        print(f"{name:28s} gate q={q:.1f}  {acc:.4f} +- {se:.4f}")

# 3. does the rule's hit rate depend on turnover or allocation volatility?
for name, v in [('MEDIAN_DAILY_TURNOVER', X_train['MEDIAN_DAILY_TURNOVER'].to_numpy()),
                ('alloc long-run vol', alloc_vol)]:
    b = pd.qcut(v, 5, duplicates='drop')
    print(name)
    print(pd.Series(p == y).groupby(b, observed=True).mean().round(4).to_string(), "\n")

# 4. is the per-allocation spread real? split days in half, correlate hit rates
rng = np.random.default_rng(0)
days = X_train['TS'].unique()
half = set(rng.choice(days, len(days) // 2, replace=False))
in_a = X_train['TS'].isin(half).to_numpy()
hit = pd.Series(p == y, index=X_train.index)
a = hit[in_a].groupby(X_train.loc[in_a, 'ALLOCATION']).mean()
b = hit[~in_a].groupby(X_train.loc[~in_a, 'ALLOCATION']).mean()
print("split-half corr of per-allocation hit rate:",
      pd.concat([a, b], axis=1).dropna().corr().iloc[0, 1])

# 5. what is going on with SIGNED_VOLUME?
vc = [f'SIGNED_VOLUME_{i}' for i in range(1, 21)]
print("NaN frac:", X_train[vc].isna().mean().mean(), "| zero frac:", (X_train[vc] == 0).mean().mean())
print("frac > 0:", (X_train[vc] > 0).mean().mean(), "| frac < 0:", (X_train[vc] < 0).mean().mean())
print("corr RET_1 vs SIGNED_VOLUME_1:", X_train[['RET_1', 'SIGNED_VOLUME_1']].corr().iloc[0, 1])
#
print("another block")
from sklearn.model_selection import GroupKFold

y = y_train_bin.loc[X_train.index].to_numpy()
ts = X_train['TS'].to_numpy()
p = (X_train['RET_1'] > 0).astype(int).to_numpy()

def per_day_acc(ts, y, pred):
    s = pd.Series(y == pred).groupby(ts).mean()
    return s.mean(), s.std() / np.sqrt(len(s))

def paired_vs_rule(ts, y, p_model, p_rule):
    diff = pd.Series((y == p_model).astype(float) - (y == p_rule).astype(float)).groupby(ts).mean()
    return diff.mean(), diff.std() / np.sqrt(len(diff))

# allocation volatility from RET windows only (no labels), so train+test together is legitimate
both = pd.concat([X_train[['ALLOCATION'] + RET_features],
                  X_test[['ALLOCATION'] + RET_features]])
alloc_vol_map = pd.Series(
    np.nanstd(both[RET_features].to_numpy(dtype=float), axis=1)
).groupby(both['ALLOCATION'].astype(str).to_numpy()).mean()

def lean_features(df):
    r = df[RET_features].to_numpy(dtype=float)       # column 0 = RET_1
    av = df['ALLOCATION'].astype(str).map(alloc_vol_map).to_numpy(dtype=float)
    return pd.DataFrame({
        'Z_EX':      r[:, 0] / (np.nanstd(r[:, 1:], axis=1) + 1e-12),
        'Z_ALLOC':   r[:, 0] / (av + 1e-12),
        'RET_1':     r[:, 0],
        'ALLOC_VOL': av,
        'TURNOVER':  df['MEDIAN_DAILY_TURNOVER'].to_numpy(dtype=float),
        'RET_4':     r[:, 3],
        'RET_8':     r[:, 7],
    }, index=df.index)

lean = lean_features(X_train)
lean_test = lean_features(X_test)

lean_params = dict(objective='binary', learning_rate=0.01, num_leaves=4,
                   min_data_in_leaf=2000, lambda_l2=50, feature_fraction=0.8,
                   bagging_fraction=0.7, bagging_freq=1, verbose=-1, num_threads=8)

def run_lean(cols, n_splits=5, seed=0):
    oof = np.full(len(lean), np.nan)
    for tr, va in GroupKFold(n_splits=n_splits).split(lean, groups=ts):
        mdl = lgbm.train({**lean_params, 'seed': seed},
                         lgbm.Dataset(lean.iloc[tr][cols], label=y[tr]),
                         num_boost_round=300)
        oof[va] = mdl.predict(lean.iloc[va][cols])
    return oof

# ablation, each variant scored against the RET_1 > 0 rule
all_cols = list(lean.columns)
variants_cols = {
    'all 7': all_cols,
    'no RET_4/RET_8': [c for c in all_cols if c not in ('RET_4', 'RET_8')],
    'no TURNOVER/ALLOC_VOL': [c for c in all_cols if c not in ('TURNOVER', 'ALLOC_VOL')],
    'Z_ALLOC + RET_1 only': ['Z_ALLOC', 'RET_1'],
}
for name, cols in variants_cols.items():
    pm = (run_lean(cols) > 0.5).astype(int)
    print(f"{name:26s} acc {per_day_acc(ts, y, pm)[0]:.4f}   vs rule {paired_vs_rule(ts, y, pm, p)}")

# is the lag-1 volume gap per day, per allocation, or neither? does it matter for the rule?
v1_nan = X_train['SIGNED_VOLUME_1'].isna()
print("per-day NaN fraction:\n", v1_nan.groupby(X_train['TS']).mean().describe())
print("per-allocation NaN fraction:\n", v1_nan.groupby(X_train['ALLOCATION']).mean().describe())
print("rule hit rate by lag-1 volume missing:\n",
      pd.Series(p == y).groupby(v1_nan.to_numpy()).agg(['mean', 'size']))
#minimal model
from sklearn.model_selection import GroupKFold

lean = pd.DataFrame({
    'Z_EX':      variants['RET_1 / std(2..20)'],
    'Z_ALLOC':   variants['RET_1 / alloc long-run vol'],
    'RET_1':     ret1,
    'ALLOC_VOL': alloc_vol,
    'TURNOVER':  X_train['MEDIAN_DAILY_TURNOVER'].to_numpy(),
    'RET_4':     r[:, 3],
    'RET_8':     r[:, 7],
}, index=X_train.index)

lean_params = dict(objective='binary', learning_rate=0.01, num_leaves=4,
                   min_data_in_leaf=2000, lambda_l2=50, feature_fraction=0.8,
                   bagging_fraction=0.7, bagging_freq=1, verbose=-1, num_threads=8)

lean_oof = np.full(len(lean), np.nan)
for tr, va in GroupKFold(n_splits=5).split(lean, groups=ts):
    mdl = lgbm.train(lean_params, lgbm.Dataset(lean.iloc[tr], label=y[tr]), num_boost_round=300)
    lean_oof[va] = mdl.predict(lean.iloc[va])

p_lean = (lean_oof > 0.5).astype(int)
print("lean model:", per_day_acc(ts, y, p_lean))
print("lean minus RET_1>0 rule:", paired_vs_rule(ts, y, p_lean, p))

def lean_features(df, alloc_vol_map):
    r = df[RET_features].to_numpy(dtype=float)       # column 0 = RET_1
    av = df['ALLOCATION'].astype(str).map(alloc_vol_map).to_numpy(dtype=float)
    return pd.DataFrame({
        'Z_EX':      r[:, 0] / (np.nanstd(r[:, 1:], axis=1) + 1e-12),
        'Z_ALLOC':   r[:, 0] / (av + 1e-12),
        'RET_1':     r[:, 0],
        'ALLOC_VOL': av,
        'TURNOVER':  df['MEDIAN_DAILY_TURNOVER'].to_numpy(dtype=float),
        'RET_4':     r[:, 3],
        'RET_8':     r[:, 7],
    }, index=df.index)

both = pd.concat([X_train[['ALLOCATION'] + RET_features],
                  X_test[['ALLOCATION'] + RET_features]])
alloc_vol_map = pd.Series(
    np.nanstd(both[RET_features].to_numpy(dtype=float), axis=1)
).groupby(both['ALLOCATION'].astype(str).to_numpy()).mean()

# rebuild the training features through the same function (re-run the ablation if numbers shift)
lean = lean_features(X_train, alloc_vol_map)
lean_test = lean_features(X_test, alloc_vol_map)

test_prob = np.zeros(len(lean_test))
for seed in range(5):
    mdl = lgbm.train({**lean_params, 'seed': seed},
                     lgbm.Dataset(lean, label=y), num_boost_round=300)
    test_prob += mdl.predict(lean_test) / 5

lean_sub = pd.DataFrame((test_prob > 0.5).astype(int),
                        index=X_test.index, columns=['target'])
lean_sub.to_csv('outputs/lean_lgbm_predictions.csv')
cols_final = [c for c in lean.columns if c not in ('RET_4', 'RET_8')]

test_prob = np.zeros(len(lean_test))
for seed in range(5):
    mdl = lgbm.train({**lean_params, 'seed': seed},
                     lgbm.Dataset(lean[cols_final], label=y),
                     num_boost_round=300)
    test_prob += mdl.predict(lean_test[cols_final]) / 5

lean_sub = pd.DataFrame((test_prob > 0.5).astype(int),
                        index=X_test.index, columns=['target'])
assert lean_sub.index.equals(sample_submission.index), "index mismatch with sample_submission"
lean_sub.to_csv('outputs/lean_lgbm_predictions.csv')
print("fraction predicted up:", lean_sub['target'].mean())
# 1. submit the pure rule (nothing fitted), as a clean experiment
rule_sub = pd.DataFrame((X_test['RET_1'] > 0).astype(int).to_numpy(),
                        index=X_test.index, columns=['target'])
assert rule_sub.index.equals(sample_submission.index)
rule_sub.to_csv('outputs/rule_ret1_predictions.csv')
print("fraction up:", rule_sub['target'].mean())

# 2. is the test set from a different "period"? compare TURNOVER per allocation
key_tr = X_train['ALLOCATION'].astype(str)
g = X_train.groupby(key_tr)['MEDIAN_DAILY_TURNOVER']
lo, hi = g.quantile(0.05), g.quantile(0.95)
key_te = X_test['ALLOCATION'].astype(str)
t = X_test['MEDIAN_DAILY_TURNOVER'].to_numpy()
inside = (t >= key_te.map(lo).to_numpy()) & (t <= key_te.map(hi).to_numpy())
print("test rows inside train 5-95% turnover range:", np.nanmean(inside), "(about 0.9 if same period)")