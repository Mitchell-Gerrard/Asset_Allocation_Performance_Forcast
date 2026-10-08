"""Date-fold LightGBM parameter sweep for the allocation classifier.

This reuses the feature construction and Bayesian encoding from no_sweep_bay.py,
but evaluates candidates with date-level folds and writes a separate submission.
"""
from itertools import product
from pathlib import Path
import runpy

import lightgbm as lgbm
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "src" / "no_sweep_bay.py"

# Reuse the existing feature pipeline. Its final baseline run also completes,
# then this script performs the parameter sweep using the corrected date folds.
state = runpy.run_path(str(BASELINE))

X_train_minimal = state["X_train_minimal"]
X_test = state["X_test"]
y_train_bin = state["y_train_bin"]
minimal_features = state["minimal_features"]
compute_bayesian_alloc_rate = state["compute_bayesian_alloc_rate"]
sample_submission = state["sample_submission"]

train_dates = X_train_minimal["TS"].unique()
n_splits = 16
num_threads = 8
threshold = 0.5

# Broad grid for a real second-pass search. The baseline's learning rate,
# sampling fractions, tree size, and round count are represented; max_depth
# and min_child_samples remain fixed for this sweep to limit runtime.
grid = product(
    [0.003, 0.01, 0.03],       # learning_rate
    [7, 15, 31, 63, 127],      # num_leaves
    [0.3, 0.5, 0.7, 0.9],      # feature_fraction
    [0.5, 0.7, 0.9, 1.0],      # bagging_fraction
    [250, 500, 1000, 1500],    # num_boost_round
)

fold_indices = []
date_splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
for train_date_idx, val_date_idx in date_splitter.split(train_dates):
    train_dates_fold = train_dates[train_date_idx]
    val_dates_fold = train_dates[val_date_idx]
    train_row_idx = X_train_minimal.index[
        X_train_minimal["TS"].isin(train_dates_fold)
    ]
    val_row_idx = X_train_minimal.index[
        X_train_minimal["TS"].isin(val_dates_fold)
    ]
    fold_indices.append((train_row_idx, val_row_idx))


def evaluate_candidate(params, num_boost_round):
    scores = []
    for train_row_idx, val_row_idx in fold_indices:
        train_df = X_train_minimal.loc[train_row_idx].copy()
        val_df = X_train_minimal.loc[val_row_idx].copy()

        encoding, _, _, global_rate = compute_bayesian_alloc_rate(
            y_train_bin.loc[train_row_idx],
            train_df["ALLOCATION"],
            prior_strength=51,
        )
        train_df["ALLOC_bayesian_ENCODING"] = (
            train_df["ALLOCATION"].map(encoding).fillna(global_rate)
        )
        val_df["ALLOC_bayesian_ENCODING"] = (
            val_df["ALLOCATION"].map(encoding).fillna(global_rate)
        )
        train_df["RET_1_X_ALLOC"] = (
            train_df["RET_1"] * train_df["ALLOC_bayesian_ENCODING"]
        )
        val_df["RET_1_X_ALLOC"] = (
            val_df["RET_1"] * val_df["ALLOC_bayesian_ENCODING"]
        )

        model = lgbm.train(
            params,
            lgbm.Dataset(
                train_df[minimal_features],
                label=y_train_bin.loc[train_row_idx].values,
            ),
            num_boost_round=num_boost_round,
        )
        predictions = model.predict(val_df[minimal_features])
        scores.append(
            accuracy_score(
                y_train_bin.loc[val_row_idx].values,
                (predictions >= threshold).astype(int),
            )
        )

    return float(np.mean(scores)), float(np.std(scores))


results = []
for candidate_number, values in enumerate(grid, 1):
    learning_rate, num_leaves, feature_fraction, bagging_fraction, rounds = values
    params = {
        "objective": "binary",
        "metric": "binary_error",
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "max_depth": -1,
        "min_child_samples": 30,
        "feature_fraction": feature_fraction,
        "bagging_fraction": bagging_fraction,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": 42,
        "num_threads": num_threads,
    }
    mean_score, std_score = evaluate_candidate(params, rounds)
    result = {
        "candidate": candidate_number,
        "mean_accuracy": mean_score,
        "std_accuracy": std_score,
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "feature_fraction": feature_fraction,
        "bagging_fraction": bagging_fraction,
        "num_boost_round": rounds,
    }
    results.append(result)
    print(
        f"{candidate_number:02d}: accuracy={mean_score:.4f} "
        f"std={std_score:.4f} leaves={num_leaves} lr={learning_rate} "
        f"ff={feature_fraction} bf={bagging_fraction} rounds={rounds}"
    )

results_df = pd.DataFrame(results).sort_values(
    ["mean_accuracy", "std_accuracy"], ascending=[False, True]
)
results_path = ROOT / "outputs" / "datefold_parameter_sweep.csv"
results_df.to_csv(results_path, index=False)

best = results_df.iloc[0]
best_params = {
    "objective": "binary",
    "metric": "binary_error",
    "learning_rate": best["learning_rate"],
    "num_leaves": int(best["num_leaves"]),
    "max_depth": -1,
    "min_child_samples": 30,
    "feature_fraction": best["feature_fraction"],
    "bagging_fraction": best["bagging_fraction"],
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": 42,
    "num_threads": num_threads,
}

full_model = lgbm.train(
    best_params,
    lgbm.Dataset(X_train_minimal[minimal_features], label=y_train_bin.values),
    num_boost_round=int(best["num_boost_round"]),
)
full_predictions = full_model.predict(X_test[minimal_features])
submission = pd.DataFrame(
    (full_predictions >= threshold).astype(int),
    index=sample_submission.index,
    columns=["target"],
)
submission_path = ROOT / "outputs" / "predictions_lgbm_datefold_best_sweep.csv"
submission.to_csv(submission_path, index=True)

print(f"\nBest candidate: {best.to_dict()}")
print(f"Saved sweep results: {results_path}")
print(f"Saved submission: {submission_path}")
