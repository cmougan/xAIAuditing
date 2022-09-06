# %%
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, r2_score
from xgboost import XGBClassifier, XGBRegressor
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from fairtools.detector import shap_detector
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from collections import defaultdict

plt.style.use("seaborn-whitegrid")
# %%
N = 5_000
for sigma in np.linspace(0, 1, 10):
    x1 = np.random.normal(1, 1, size=N)
    x2 = np.random.normal(1, 1, size=N)
    x34 = np.random.multivariate_normal([1, 1], [[1, sigma], [sigma, 1]], size=N)
    x3 = x34[:, 0]
    x4 = x34[:, 1]
    # Binarize protected attribute
    x4 = np.where(x4 > np.mean(x4), 1, 0)

    X = pd.DataFrame([x1, x2, x3, x4]).T
    X.columns = ["var%d" % (i + 1) for i in range(X.shape[1])]
    Z = X.var4.values

    y = (x1 + x2 + x3) / 3
    y = 1 / (1 + np.exp(-y))

    model = LinearRegression()
    model.fit(X, y)

    plt.figure()
    sns.kdeplot(model.predict(X[X["var4"] == 0]), label="0", shade=True)
    sns.kdeplot(model.predict(X[X["var4"] == 1]), label="1", shade=True)
    plt.show()
    # %%

    # %%
