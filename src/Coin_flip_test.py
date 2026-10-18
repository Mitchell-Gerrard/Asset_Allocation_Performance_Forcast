import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


X_train = pd.read_csv('data/X_train.csv', index_col='ROW_ID')
X_test = pd.read_csv('data/X_test.csv', index_col='ROW_ID')
y_train = pd.read_csv('data/y_train.csv', index_col='ROW_ID')
y_train_bin = (y_train['target'] > 0).astype(int)

seeds = np.abs(np.random.randint(0, 100000, 10000))
corrects = []
for seed in seeds:
    np.random.seed(seed)
    answers=np.random.binomial(1, 0.5, size=len(y_train_bin))

    corrects.append(np.sum(answers == y_train_bin.values)/len(y_train_bin))
plt.hist(corrects, bins=20)
plt.savefig('outputs/random_accuracy_histogram_flipping.png')
plt.show()  
print(f"Random accuracy: {np.mean(corrects):.4%} ± {np.std(corrects):.4%}")