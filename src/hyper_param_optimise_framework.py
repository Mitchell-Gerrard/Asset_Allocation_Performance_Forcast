import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgbm
import optuna

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


# ============================================================
# LOAD DATA
# ============================================================

X_train = pd.read_csv('data/X_train.csv', index_col='ROW_ID')
X_test = pd.read_csv('data/X_test.csv', index_col='ROW_ID')
y_train = pd.read_csv('data/y_train.csv', index_col='ROW_ID')
sample_submission = pd.read_csv('data/sample_submission.csv', index_col='ROW_ID')

total_data = pd.concat([X_train, y_train], axis=1)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

RET_features = [f'RET_{i}' for i in range(1, 21)]
SIGNED_VOLUME_features = [f'SIGNED_VOLUME_{i}' for i in range(1, 21)]
TURNOVER_features = ['MEDIAN_DAILY_TURNOVER']


# Cross-sectional average return
X_train['CROSS_SECTIONAL_AVG_RET1'] = (
    total_data.groupby('TS')['RET_1'].transform('mean')
)

X_test['CROSS_SECTIONAL_AVG_RET1'] = (
    X_test.groupby('TS')['RET_1'].transform('mean')
)


# Average performance
for i in [3, 5, 10, 15, 20]:

    X_train[f'AVERAGE_PERF_{i}'] = X_train[RET_features[:i]].mean(1)

    X_train[f'ALLOCATIONS_AVERAGE_PERF_{i}'] = (
        X_train.groupby('TS')[f'AVERAGE_PERF_{i}'].transform('mean')
    )

    X_test[f'AVERAGE_PERF_{i}'] = X_test[RET_features[:i]].mean(1)

    X_test[f'ALLOCATIONS_AVERAGE_PERF_{i}'] = (
        X_test.groupby('TS')[f'AVERAGE_PERF_{i}'].transform('mean')
    )


# Standard deviation of performance
for i in [20]:

    X_train[f'STD_PERF_{i}'] = X_train[RET_features[:i]].std(1)

    X_train[f'ALLOCATIONS_STD_PERF_{i}'] = (
        X_train.groupby('TS')[f'STD_PERF_{i}'].transform('mean')
    )

    X_test[f'STD_PERF_{i}'] = X_test[RET_features[:i]].std(1)

    X_test[f'ALLOCATIONS_STD_PERF_{i}'] = (
        X_test.groupby('TS')[f'STD_PERF_{i}'].transform('mean')
    )


features = RET_features + SIGNED_VOLUME_features + TURNOVER_features

features += [
    f'AVERAGE_PERF_{i}'
    for i in [3, 5, 10, 15, 20]
]

features += [
    f'ALLOCATIONS_AVERAGE_PERF_{i}'
    for i in [3, 5, 10, 15, 20]
]

features += [
    f'STD_PERF_{i}'
    for i in [20]
]

features += [
    f'ALLOCATIONS_STD_PERF_{i}'
    for i in [20]
]


minimal_features = features + [
    'CROSS_SECTIONAL_AVG_RET1',
    'ALLOC_TARGET_ENCODING'
]


# ============================================================
# VOLUME FEATURES
# ============================================================

SIGNED_VOLUME_features_clean = [
    f'SIGNED_VOLUME_{i}'
    for i in range(2, 21)
]

for i in [3, 5, 10, 15, 20]:

    cols = SIGNED_VOLUME_features_clean[:i - 1]

    X_train[f'AVERAGE_VOLUME_{i}'] = X_train[cols].mean(1)

    X_train[f'ALLOCATIONS_AVERAGE_VOLUME_{i}'] = (
        X_train.groupby('TS')[f'AVERAGE_VOLUME_{i}'].transform('mean')
    )

    X_test[f'AVERAGE_VOLUME_{i}'] = X_test[cols].mean(1)

    X_test[f'ALLOCATIONS_AVERAGE_VOLUME_{i}'] = (
        X_test.groupby('TS')[f'AVERAGE_VOLUME_{i}'].transform('mean')
    )


for i in [20]:

    cols = SIGNED_VOLUME_features_clean[:i - 1]

    X_train[f'STD_VOLUME_{i}'] = X_train[cols].std(1)

    X_train[f'ALLOCATIONS_STD_VOLUME_{i}'] = (
        X_train.groupby('TS')[f'STD_VOLUME_{i}'].transform('mean')
    )

    X_test[f'STD_VOLUME_{i}'] = X_test[cols].std(1)

    X_test[f'ALLOCATIONS_STD_VOLUME_{i}'] = (
        X_test.groupby('TS')[f'STD_VOLUME_{i}'].transform('mean')
    )


volume_features = (
    [
        f'AVERAGE_VOLUME_{i}'
        for i in [3, 5, 10, 15, 20]
    ]
    +
    [
        f'ALLOCATIONS_AVERAGE_VOLUME_{i}'
        for i in [3, 5, 10, 15, 20]
    ]
    +
    [
        f'STD_VOLUME_{i}'
        for i in [20]
    ]
    +
    [
        f'ALLOCATIONS_STD_VOLUME_{i}'
        for i in [20]
    ]
)

features += volume_features

minimal_features = features + [
    'CROSS_SECTIONAL_AVG_RET1',
    'ALLOC_TARGET_ENCODING'
]


# ============================================================
# CROSS-VALIDATION SETUP
# ============================================================

X_train_minimal = X_train.copy()

train_dates = X_train_minimal['TS'].unique()

n_splits = 8

kf = KFold(
    n_splits=n_splits,
    random_state=0,
    shuffle=True
)

splits = list(kf.split(train_dates))


# ============================================================
# PREPARE THE 8 CV FOLDS ONCE
#
# This is important because otherwise Optuna would repeatedly
# perform the target encoding work for every trial.
# ============================================================

fold_data = []

for fold, (local_train_dates_ids, local_test_dates_ids) in enumerate(splits):

    local_train_dates = train_dates[local_train_dates_ids]
    local_test_dates = train_dates[local_test_dates_ids]

    local_train_ids = X_train_minimal['TS'].isin(local_train_dates)
    local_test_ids = X_train_minimal['TS'].isin(local_test_dates)

    # --------------------------------------------------------
    # Target encoding using ONLY training rows in this fold
    # --------------------------------------------------------

    fold_train_targets = y_train.loc[
        local_train_ids, 'target'
    ]

    fold_train_allocs = X_train_minimal.loc[
        local_train_ids, 'ALLOCATION'
    ]

    alloc_encoding = (
        pd.DataFrame({
            'ALLOCATION': fold_train_allocs,
            'target': fold_train_targets.values
        })
        .groupby('ALLOCATION')['target']
        .apply(lambda s: (np.sign(s) > 0).mean())
    )

    global_rate = (
        (np.sign(fold_train_targets) > 0).mean()
    )

    # Create fold-specific copies
    X_fold_train = X_train_minimal.loc[
        local_train_ids
    ].copy()

    X_fold_test = X_train_minimal.loc[
        local_test_ids
    ].copy()

    X_fold_train['ALLOC_TARGET_ENCODING'] = (
        X_fold_train['ALLOCATION']
        .map(alloc_encoding)
        .fillna(global_rate)
    )

    X_fold_test['ALLOC_TARGET_ENCODING'] = (
        X_fold_test['ALLOCATION']
        .map(alloc_encoding)
        .fillna(global_rate)
    )

    y_fold_train = y_train.loc[
        local_train_ids, 'target'
    ].copy()

    y_fold_test = y_train.loc[
        local_test_ids, 'target'
    ].copy()

    fold_data.append(
        (
            X_fold_train[minimal_features],
            y_fold_train,
            X_fold_test[minimal_features],
            y_fold_test
        )
    )

    print(
        f"Prepared fold {fold + 1}/{n_splits}"
    )


# ============================================================
# OPTUNA OBJECTIVE
# ============================================================

def objective(trial):

    params = {
        'objective': 'mse',
        'metric': 'mse',

        'verbosity': -1,

        # Keep this below your total CPU count if necessary.
        'num_threads': 50,

        'seed': 42,

        # -------------------------
        # Main hyperparameters
        # -------------------------

        'learning_rate': trial.suggest_float(
            'learning_rate',
            0.003,
            0.1,
            log=True
        ),

        'num_leaves': trial.suggest_int(
            'num_leaves',
            8,
            128
        ),

        'max_depth': trial.suggest_int(
            'max_depth',
            2,
            12
        ),

        'min_child_samples': trial.suggest_int(
            'min_child_samples',
            5,
            100
        ),

        # -------------------------
        # Random feature / row sampling
        # -------------------------

        'feature_fraction': trial.suggest_float(
            'feature_fraction',
            0.6,
            1.0
        ),

        'bagging_fraction': trial.suggest_float(
            'bagging_fraction',
            0.6,
            1.0
        ),

        'bagging_freq': 1,

        # -------------------------
        # Regularisation
        # -------------------------

        'lambda_l1': trial.suggest_float(
            'lambda_l1',
            1e-8,
            10.0,
            log=True
        ),

        'lambda_l2': trial.suggest_float(
            'lambda_l2',
            1e-8,
            10.0,
            log=True
        ),
    }

    fold_scores = []

    for fold, (
        X_local_train,
        y_local_train,
        X_local_test,
        y_local_test
    ) in enumerate(fold_data):

        train_data = lgbm.Dataset(
            X_local_train,
            label=y_local_train.values
        )

        valid_data = lgbm.Dataset(
            X_local_test,
            label=y_local_test.values,
            reference=train_data
        )

        model = lgbm.train(
            params,
            train_data,
            num_boost_round=2000,

            valid_sets=[valid_data],

            callbacks=[
                lgbm.early_stopping(
                    stopping_rounds=75,
                    verbose=False
                )
            ]
        )

        y_local_pred = model.predict(
            X_local_test,
            num_threads=params['num_threads']
        )

        score = accuracy_score(
            (y_local_test > 0).astype(int),
            (y_local_pred > 0).astype(int)
        )

        fold_scores.append(score)

        # Allow Optuna to stop bad trials early
        trial.report(
            np.mean(fold_scores),
            fold
        )

        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(fold_scores)


# ============================================================
# RUN OPTUNA
# ============================================================

N_TRIALS = 50

study = optuna.create_study(
    direction='maximize',
    study_name='lgbm_sign_accuracy'
)

study.optimize(
    objective,
    n_trials=N_TRIALS,
    show_progress_bar=True
)


# ============================================================
# BEST PARAMETERS
# ============================================================

print("\n" + "=" * 60)
print("OPTUNA RESULTS")
print("=" * 60)

print(f"Best CV accuracy: {study.best_value * 100:.2f}%")

print("\nBest parameters:")

for key, value in study.best_params.items():
    print(f"  {key}: {value}")


# ============================================================
# FINAL CV USING BEST PARAMETERS
# ============================================================

best_params = {
    'objective': 'mse',
    'metric': 'mse',
    'verbosity': -1,
    'num_threads': 50,
    'seed': 42,

    **study.best_params
}

final_cv_scores = []
best_iterations = []

print("\n" + "=" * 60)
print("FINAL 8-FOLD CV")
print("=" * 60)

for fold, (
    X_local_train,
    y_local_train,
    X_local_test,
    y_local_test
) in enumerate(fold_data):

    train_data = lgbm.Dataset(
        X_local_train,
        label=y_local_train.values
    )

    valid_data = lgbm.Dataset(
        X_local_test,
        label=y_local_test.values,
        reference=train_data
    )

    model = lgbm.train(
        best_params,
        train_data,
        num_boost_round=2000,

        valid_sets=[valid_data],

        callbacks=[
            lgbm.early_stopping(
                stopping_rounds=75,
                verbose=False
            )
        ]
    )

    best_iterations.append(
        model.best_iteration
    )

    y_local_pred = model.predict(
        X_local_test,
        num_threads=best_params['num_threads']
    )

    score = accuracy_score(
        (y_local_test > 0).astype(int),
        (y_local_pred > 0).astype(int)
    )

    final_cv_scores.append(score)

    print(
        f"Fold {fold + 1} - "
        f"Accuracy: {score * 100:.2f}% - "
        f"Trees: {model.best_iteration}"
    )


mean = np.mean(final_cv_scores) * 100
std = np.std(final_cv_scores) * 100

print(
    f"\nOptimised CV accuracy: "
    f"{mean:.2f}% (+- {std:.2f})"
)

print(
    f"Average best iteration: "
    f"{np.mean(best_iterations):.0f}"
)


# ============================================================
# FINAL MODEL
#
# Re-create target encoding using the FULL training set
# ============================================================

full_alloc_encoding = (
    total_data
    .groupby('ALLOCATION')['target']
    .apply(lambda s: (np.sign(s) > 0).mean())
)

full_global_rate = (
    (np.sign(total_data['target']) > 0).mean()
)


# Test target encoding
X_test['ALLOC_TARGET_ENCODING'] = (
    X_test['ALLOCATION']
    .map(full_alloc_encoding)
    .fillna(full_global_rate)
)


# Train target encoding
X_train_minimal['ALLOC_TARGET_ENCODING'] = (
    X_train_minimal['ALLOCATION']
    .map(full_alloc_encoding)
    .fillna(full_global_rate)
)


# ============================================================
# FINAL TRAINING
#
# Use the average optimal number of boosting rounds found
# during CV.
# ============================================================

FINAL_NUM_BOOST_ROUND = max(
    1,
    int(round(np.mean(best_iterations)))
)

print(
    f"\nTraining final model with "
    f"{FINAL_NUM_BOOST_ROUND} boosting rounds..."
)

final_train_data = lgbm.Dataset(
    X_train_minimal[minimal_features],
    label=y_train['target'].values
)

final_model = lgbm.train(
    best_params,
    final_train_data,
    num_boost_round=FINAL_NUM_BOOST_ROUND
)


# ============================================================
# PREDICTIONS
# ============================================================

preds_lgbm = final_model.predict(
    X_test[minimal_features],
    num_threads=best_params['num_threads']
)

preds_lgbm = pd.DataFrame(
    preds_lgbm,
    index=sample_submission.index,
    columns=['target']
)


# ============================================================
# SAVE SUBMISSION
# ============================================================

output_path = (
    'outputs/'
    'predictions_lgbm_optuna_allocations_cross_volume_plus_baseline.csv'
)

(
    (preds_lgbm > 0)
    .astype(int)
    .to_csv(output_path, index=True)
)

print(f"\nSaved predictions to: {output_path}")
