import kpg
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

prefix = "pickles/"
file = prefix + "2-25-5-cij"

with open(file + ".pickle", "rb") as f:
    results = pickle.load(f)

lines = []
opt_vals = []

for key, value in results.items():
    n = value.kpg.n
    m = value.kpg.m
    combis = (n - 1) + (n - 2)
    interactions = np.zeros((combis, m))
    i = 0
    for p1, p2 in value.kpg.pairs:
        if p1 > p2:
            continue
        interactions[i, :] = np.multiply(value.X[p1, :], value.X[p2, :])
        i += 1
    lines.append(np.concatenate((value.X.flatten(), interactions.flatten())))
    opt_vals.append(value.ObjVal)

X = pd.DataFrame(lines)
y = pd.DataFrame(opt_vals)

lr = LinearRegression(fit_intercept=False)
lr.fit(X, y)
coefs = np.reshape(lr.coef_, (-1, m))
print(coefs)
print(results["master"].kpg)
