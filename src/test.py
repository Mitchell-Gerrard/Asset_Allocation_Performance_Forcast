prev_scores = [0.5192, 0.5219, 0.5171, 0.5213, 0.5181, 0.5213, 0.5230, 0.5225]  # your RET_1+turnover-only run
new_scores = [0.5217, 0.5254, 0.5187, 0.5255, 0.5202, 0.5244, 0.5268, 0.5237]  # this run

improvements = [n - p for n, p in zip(new_scores, prev_scores)]
print(f"Per-fold improvement: {[f'{d*100:+.2f}pp' for d in improvements]}")
print(f"Folds improved: {sum(1 for d in improvements if d > 0)}/8")