# %%
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor
from fairtools.detector import ExplanationAudit

# %%
N = 10_000
x1 = np.random.normal(1, 1, size=N)
x2 = np.random.normal(1, 1, size=N)
x34 = np.random.multivariate_normal([1, 1], [[1, 0.5], [0.5, 1]], size=N)
x3 = x34[:, 0]
x4 = x34[:, 1]
# Binarize protected attribute
x4 = np.where(x4 > np.mean(x4), 1, 0)
X = pd.DataFrame([x1, x2, x3, x4]).T
X.columns = ["var%d" % (i + 1) for i in range(X.shape[1])]
y = (x1 + x2 + x3) / 3
y = 1 / (1 + np.exp(-y))
detector = ExplanationAudit(model=XGBRegressor(), gmodel=LogisticRegression())
detector.fit(X, y, Z="var4")
detector.get_auc_val()
# %%
coefs = detector.gmodel.coef_[0]
coefs = pd.DataFrame(coefs, index=X.columns[:-1], columns=["coef"]).sort_values(
    "coef", ascending=False
)
coefs.plot(kind="bar")
# %%
import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")
plt.figure(figsize=(10, 10))
coefs.plot(kind="bar")
plt.tight_layout()
plt.savefig("images/coefs_synth.png")
plt.show()

# %%
