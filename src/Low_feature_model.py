import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgbm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
X_train = pd.read_csv('data/X_train.csv',index_col='ROW_ID')
X_test = pd.read_csv('data/X_test.csv',index_col='ROW_ID')
y_train = pd.read_csv('data/y_train.csv',index_col='ROW_ID')
sample_submission = pd.read_csv('data/sample_submission.csv',index_col='ROW_ID')
total_data = pd.concat([X_train,y_train],axis=1)

minimal_features = ['RET_1']

X_train_minimal = X_train.copy()


lgbm_params = {
"objective": "binary",
"metric": "binary_logloss",
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

    X_local_train = X_train_minimal.loc[local_train_ids, minimal_features]
    y_local_train = (y_train.loc[local_train_ids, 'target'] > 0).astype(int)

    X_local_test = X_train_minimal.loc[local_test_ids, minimal_features]
    y_local_test = y_train.loc[local_test_ids, 'target']

    train_data = lgbm.Dataset(X_local_train, label=y_local_train.values)

    model = lgbm.train(lgbm_params, train_data, num_boost_round=NUM_BOOST_ROUND)
    y_local_pred = model.predict(X_local_test, num_threads=lgbm_params['num_threads'])

    score = accuracy_score((y_local_test > 0).astype(int), (y_local_pred > 0.5).astype(int))
    scores_minimal.append(score)
    print(f"Fold {i+1} - Accuracy: {score * 100:.2f}%")

mean = np.mean(scores_minimal) * 100
std = np.std(scores_minimal) * 100
print(f"\nMinimal model accuracy: {mean:.2f}% (+- {std:.2f})")
preds_lgbm = model.predict(X_test[minimal_features], num_threads=lgbm_params['num_threads'])
preds_lgbm = pd.DataFrame(preds_lgbm, index=sample_submission.index,columns=['target']) 
(preds_lgbm > 0.5).astype(int).to_csv('outputs/predictions_lgbm_minimal.csv', index=True)