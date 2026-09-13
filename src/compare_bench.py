import pandas as pd
import numpy as np



lgbm_bench = pd.read_csv('data/preds_lgbm_bench.csv', index_col='ROW_ID')
ridge_bench = pd.read_csv('data/preds_ridge_bench.csv', index_col='ROW_ID')
My_model_minimal = pd.read_csv('outputs/predictions_lgbm_minimal.csv', index_col='ROW_ID')
My_model_allocations = pd.read_csv('outputs/predictions_lgbm_allocations.csv', index_col='ROW_ID')

def compare_predictions(bench_df, yours_df, label):
    comparison = bench_df.join(yours_df, lsuffix='_bench', rsuffix='_yours')

    agreement_rate = (comparison['target_bench'] == comparison['target_yours']).mean() * 100
    print(f"[{label}] Agreement rate with benchmark: {agreement_rate:.2f}%")

    print(f"Benchmark predicted-positive rate: {(comparison['target_bench'] == 1).mean() * 100:.2f}%")
    print(f"Your model predicted-positive rate: {(comparison['target_yours'] == 1).mean() * 100:.2f}%")

    disagreements = comparison[comparison['target_bench'] != comparison['target_yours']]
    print(f"Disagreements: {len(disagreements)} rows ({len(disagreements)/len(comparison)*100:.2f}%)")
    print(disagreements['target_bench'].value_counts())
    print()

compare_predictions(lgbm_bench, My_model_minimal, "minimal (RET_1 + turnover) vs lgbm_bench")
compare_predictions(lgbm_bench, My_model_allocations, "allocations model vs lgbm_bench")