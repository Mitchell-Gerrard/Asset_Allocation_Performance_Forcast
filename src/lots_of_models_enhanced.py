import numpy as np
import pandas as pd
import lightgbm as lgbm
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier


try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("xgboost not installed -- skipping. `pip install xgboost` to include it.")

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
# Each entry: name -> (train_fn, predict_fn, weight)
# train_fn(X, y) -> fitted model object
# predict_fn(model, X) -> continuous scores, positive = predict long
# Set weight to 0 to disable a model without deleting its code.
# ============================================================

def train_lgbm(X, y):
    lgbm_params = {
        "objective": "mse", "metric": "mse", "num_threads": 50, "seed": 42,
        "verbosity": -1, 'learning_rate': 1e-2, 'max_depth': 3,
    }
    train_data = lgbm.Dataset(X, label=y.values)
    return lgbm.train(lgbm_params, train_data, num_boost_round=500)

def predict_lgbm(model, X):
    return model.predict(X, num_threads=50)


def train_ridge(X, y):
    model = Ridge(alpha=1e-2, fit_intercept=True)
    model.fit(X.fillna(0), y)
    return model

def predict_ridge(model, X):
    return model.predict(X.fillna(0))


def train_logistic(X, y):
    # scale first -- logistic regression is sensitive to feature scale, unlike trees
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.fillna(0))
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_scaled, (y > 0).astype(int))
    return (model, scaler)

def predict_logistic(model_tuple, X):
    model, scaler = model_tuple
    X_scaled = scaler.transform(X.fillna(0))
    # return probability of positive class, recentred around 0 so > 0 means "predict long"
    return model.predict_proba(X_scaled)[:, 1] - 0.5


def train_mlp(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.fillna(0))
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        alpha=1e-3,          # L2 regularization -- important given how thin the signal is
        learning_rate_init=1e-3,
        max_iter=200,
        early_stopping=True,
        random_state=42,
    )
    model.fit(X_scaled, (y > 0).astype(int))
    return (model, scaler)

def predict_mlp(model_tuple, X):
    model, scaler = model_tuple
    X_scaled = scaler.transform(X.fillna(0))
    return model.predict_proba(X_scaled)[:, 1] - 0.5


def train_xgb(X, y):
    dtrain = xgb.DMatrix(X, label=(y > 0).astype(int))
    params = {
        'objective': 'binary:logistic',
        'max_depth': 3,
        'eta': 1e-2,
        'nthread': 50,
        'seed': 42,
        'verbosity': 0,
    }
    return xgb.train(params, dtrain, num_boost_round=500)

def predict_xgb(model, X):
    dtest = xgb.DMatrix(X)
    return model.predict(dtest) - 0.5


def train_lgbm_binary(X, y):
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_threads": 50,
        "seed": 42,
        "verbosity": -1,
        "learning_rate": 1e-2,
        "max_depth": 3,
    }
    train_data = lgbm.Dataset(X, label=(y > 0).astype(int).values)
    return lgbm.train(params, train_data, num_boost_round=500)


def predict_lgbm_binary(model, X):
    return model.predict(X, num_threads=50) - 0.5


def train_extratrees(X, y):
    model = ExtraTreesClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=20,
        max_features=0.7,
        n_jobs=50,
        random_state=42,
    )
    model.fit(X.fillna(0), (y > 0).astype(int))
    return model


def predict_extratrees(model, X):
    return model.predict_proba(X.fillna(0))[:, 1] - 0.5


MODEL_REGISTRY = {
    'lgbm': (train_lgbm, predict_lgbm, 1.0),
    'lgbm_binary': (train_lgbm_binary, predict_lgbm_binary, 1.0),
    'ridge': (train_ridge, predict_ridge, 1.0),
    'logistic': (train_logistic, predict_logistic, 1.0),
    'mlp': (train_mlp, predict_mlp, 0.0),
    'extratrees': (train_extratrees, predict_extratrees, 0.0),
}
if HAS_XGB:
    MODEL_REGISTRY['xgboost'] = (train_xgb, predict_xgb, 1.0)

def get_active_models():
    return {name: cfg for name, cfg in MODEL_REGISTRY.items() if cfg[2] != 0}


def train_and_predict_all(X_local_train, y_local_train, X_local_test):
    """Train every registered model so candidate models can be compared."""
    individual_preds = {}
    for name, (train_fn, predict_fn, weight) in MODEL_REGISTRY.items():
        if name == 'xgboost' and not HAS_XGB:
            continue
        model = train_fn(X_local_train, y_local_train)
        individual_preds[name] = predict_fn(model, X_local_test)
    return individual_preds


def weighted_ensemble(individual_preds, weights=None):
    """Weighted ensemble after per-fold z-scoring."""
    if weights is None:
        weights = {name: MODEL_REGISTRY[name][2] for name in individual_preds}

    normalized_preds = {}
    for name, pred in individual_preds.items():
        pred = np.asarray(pred)
        normalized_preds[name] = (pred - pred.mean()) / (pred.std() + 1e-9)

    weighted_sum = np.zeros(len(next(iter(normalized_preds.values()))))
    total_weight = 0.0

    for name, pred in normalized_preds.items():
        weight = weights.get(name, 0.0)
        if weight > 0:
            weighted_sum += weight * pred
            total_weight += weight

    if total_weight == 0:
        raise ValueError("No positive ensemble weights.")
    return weighted_sum / total_weight


def simple_two_model_mix(individual_preds, a, b, w_a):
    """Mix two raw model scores after per-fold z-scoring."""
    pa = np.asarray(individual_preds[a])
    pb = np.asarray(individual_preds[b])
    za = (pa - pa.mean()) / (pa.std() + 1e-9)
    zb = (pb - pb.mean()) / (pb.std() + 1e-9)
    return w_a * za + (1.0 - w_a) * zb


def print_disagreement_analysis(individual_preds, y_true_binary):
    """Show which model wins when two models disagree."""
    names = list(individual_preds.keys())
    print(f"\n{'='*60}")
    print("DISAGREEMENT ANALYSIS")
    print(f"{'='*60}")

    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            pa = (np.asarray(individual_preds[names[a]]) > 0).astype(int)
            pb = (np.asarray(individual_preds[names[b]]) > 0).astype(int)
            disagree = pa != pb

            if not disagree.any():
                continue

            a_correct = (pa[disagree] == y_true_binary[disagree]).mean() * 100
            b_correct = (pb[disagree] == y_true_binary[disagree]).mean() * 100

            print(
                f"{names[a]} vs {names[b]} | "
                f"disagree: {disagree.mean()*100:.2f}% | "
                f"{names[a]} correct: {a_correct:.2f}% | "
                f"{names[b]} correct: {b_correct:.2f}%"
            )


def find_best_two_model_mix(fold_preds, fold_targets, a, b):
    """Find the best simple weight for a two-model mixture using OOF folds."""
    best = None

    for w in np.arange(0.0, 1.01, 0.05):
        fold_scores = []

        for preds, y_true in zip(fold_preds, fold_targets):
            mixed = simple_two_model_mix(preds, a, b, w)
            score = accuracy_score(y_true, (mixed > 0).astype(int))
            fold_scores.append(score)

        mean_score = float(np.mean(fold_scores))

        if best is None or mean_score > best["mean"]:
            best = {
                "weight_a": float(w),
                "weight_b": float(1.0 - w),
                "mean": mean_score,
                "std": float(np.std(fold_scores)),
            }

    return best


# ============================================================
# CROSS-VALIDATION LOOP
# ============================================================
train_dates = X_train_minimal['TS'].unique()
n_splits = 8

scores_by_model = {name: [] for name in MODEL_REGISTRY}
fold_targets = []
scores_ensemble = []
all_fold_preds = []  # keep per-fold individual predictions for agreement analysis

splits = KFold(n_splits=n_splits, random_state=0, shuffle=True).split(train_dates)

for i, (local_train_dates_ids, local_test_dates_ids) in enumerate(splits):
    local_train_dates = train_dates[local_train_dates_ids]
    local_test_dates = train_dates[local_test_dates_ids]

    local_train_ids = X_train_minimal['TS'].isin(local_train_dates)
    local_test_ids = X_train_minimal['TS'].isin(local_test_dates)

    # --- allocation target encoding, fold-scoped ---
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

    individual_preds = train_and_predict_all(X_local_train, y_local_train, X_local_test)
    ensemble_pred = weighted_ensemble(individual_preds)

    y_true_binary = (y_local_test > 0).astype(int)
    fold_line = []
    for name, pred in individual_preds.items():
        s = accuracy_score(y_true_binary, (pred > 0).astype(int))
        scores_by_model[name].append(s)
        fold_line.append(f"{name}: {s * 100:.2f}%")

    ensemble_score = accuracy_score(y_true_binary, (ensemble_pred > 0).astype(int))
    scores_ensemble.append(ensemble_score)

    print(f"Fold {i+1} - Ensemble: {ensemble_score * 100:.2f}%  |  " + "  ".join(fold_line))
    all_fold_preds.append(individual_preds)
    fold_targets.append(y_true_binary.to_numpy())

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"Ensemble accuracy: {np.mean(scores_ensemble) * 100:.2f}% (+- {np.std(scores_ensemble) * 100:.2f})")
for name, scores in scores_by_model.items():
    print(f"{name} accuracy: {np.mean(scores) * 100:.2f}% (+- {np.std(scores) * 100:.2f})")

# ============================================================
# MODEL DIVERSITY / MIXTURE ANALYSIS
# ============================================================
print(f"\n{'='*60}")
print("MODEL MIXTURE SEARCH (OOF ONLY)")
print(f"{'='*60}")

candidate_pairs = [
    ('lgbm', 'xgboost'),
    ('lgbm', 'logistic'),
    ('lgbm', 'ridge'),
]

for a, b in candidate_pairs:
    if a in all_fold_preds[0] and b in all_fold_preds[0]:
        best = find_best_two_model_mix(
            all_fold_preds,
            fold_targets,
            a,
            b
        )
        print(
            f"{a} + {b}: "
            f"{best['weight_a']:.2f}/{best['weight_b']:.2f} | "
            f"{best['mean']*100:.2f}% (+- {best['std']*100:.2f})"
        )

# Detailed disagreement analysis across every fold.
for preds, y_true in zip(all_fold_preds, fold_targets):
    print_disagreement_analysis(preds, y_true)

# ============================================================
# PAIRWISE PREDICTION AGREEMENT (ALL OOF FOLDS)
# ============================================================
print(f"\n{'='*60}")
print("PAIRWISE PREDICTION AGREEMENT (ALL OOF)")
print(f"{'='*60}")

names = list(all_fold_preds[0].keys())

for a in range(len(names)):
    for b in range(a + 1, len(names)):
        agreements = []

        for preds in all_fold_preds:
            pred_a = (preds[names[a]] > 0).astype(int)
            pred_b = (preds[names[b]] > 0).astype(int)
            agreements.append((pred_a == pred_b).mean())

        print(
            f"{names[a]} vs {names[b]}: "
            f"{np.mean(agreements)*100:.2f}% agreement"
        )


# ============================================================
# FINAL MODEL(S) FOR SUBMISSION (trained on full training set)
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

final_individual_preds = train_and_predict_all(
    X_train_minimal[minimal_features], y_train['target'], X_test[minimal_features]
)
final_ensemble_pred = weighted_ensemble(final_individual_preds)

# Best OOF-selected simple LGBM/XGBoost mixture.
best_lgbm_xgb = find_best_two_model_mix(
    all_fold_preds,
    fold_targets,
    'lgbm',
    'xgboost'
) if HAS_XGB else None

if best_lgbm_xgb is not None:
    final_lgbm_xgb_pred = simple_two_model_mix(
        final_individual_preds,
        'lgbm',
        'xgboost',
        best_lgbm_xgb['weight_a']
    )
else:
    final_lgbm_xgb_pred = None

preds_df = pd.DataFrame(final_ensemble_pred, index=sample_submission.index, columns=['target'])
(preds_df > 0).astype(int).to_csv('outputs/predictions_ensemble.csv', index=True)

for name, pred in final_individual_preds.items():
    individual_df = pd.DataFrame(pred, index=sample_submission.index, columns=['target'])
    (individual_df > 0).astype(int).to_csv(f'outputs/predictions_{name}_only.csv', index=True)

if final_lgbm_xgb_pred is not None:
    lgbm_xgb_df = pd.DataFrame(
        final_lgbm_xgb_pred,
        index=sample_submission.index,
        columns=['target']
    )
    (lgbm_xgb_df > 0).astype(int).to_csv(
        'outputs/predictions_lgbm_xgb_mix.csv',
        index=True
    )

print("\nSaved: outputs/predictions_ensemble.csv, predictions_lgbm_xgb_mix.csv, and one file per model.")