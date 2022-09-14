# %%
from cmath import exp
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
from scipy.stats import wasserstein_distance

plt.style.use("seaborn-whitegrid")
# %%
N = 5_000
dp = []
res = []
exp_evolution = pd.DataFrame()
for gamma in np.linspace(0, 1, 10):
    x1 = np.random.normal(1, 1, size=N)
    x2 = np.random.normal(1, 1, size=N)
    x34 = np.random.multivariate_normal([1, 1], [[1, gamma], [gamma, 1]], size=N)
    x3 = x34[:, 0]
    x4 = x34[:, 1]
    # Binarize protected attribute
    x4 = np.where(x4 > np.mean(x4), 1, 0)

    X = pd.DataFrame([x1, x2, x3, x4]).T
    X.columns = ["var%d" % (i + 1) for i in range(X.shape[1])]

    y = (x1 + x2 + x3) / 3
    y = 1 / (1 + np.exp(-y))

    # Train test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=0.33,
        random_state=42,
    )
    Z_tr = X_tr.var4.values
    Z_te = X_te.var4.values

    # Drop protected attribute
    X_tr = X_tr.drop("var4", axis=1)
    X_te = X_te.drop("var4", axis=1)

    # Learning of the model F
    model = LinearRegression()
    model.fit(X_tr, y_tr)

    preds_tr = model.predict(X_tr)
    preds_te = model.predict(X_te)

    # Demographic parity calculation
    dist = wasserstein_distance(
        model.predict(X_te[Z_te == 0]), model.predict(X_te[Z_te == 1])
    )
    dp.append(dist)
    # Visualizations of distribution of predictions
    # plt.figure()
    # sns.kdeplot(model.predict(X_te[Z_te == 0]), label="0", shade=True)
    # sns.kdeplot(model.predict(X_te[Z_te == 1]), label="1", shade=True)
    # plt.show()

    # SHAP
    explainer = shap.LinearExplainer(
        model, X_tr, feature_dependence="correlation_dependent"
    )
    # explainer = shap.Explainer(model)
    shapX1 = explainer(X_tr).values
    shapX1 = pd.DataFrame(shapX1)
    shapX1.columns = ["var%d" % (i + 1) for i in range(shapX1.shape[1])]
    shapX2 = explainer(X_te).values
    shapX2 = pd.DataFrame(shapX2)
    shapX2.columns = ["var%d" % (i + 1) for i in range(shapX2.shape[1])]

    m = LogisticRegression()
    m.fit(shapX1, Z_tr)
    res1 = roc_auc_score(Z_te, m.predict_proba(shapX2)[:, 1])
    exp_evolution[gamma] = pd.DataFrame(m.coef_.squeeze())

    # Output
    m = LogisticRegression()
    m.fit(preds_tr.reshape(-1, 1), Z_tr)
    res2 = roc_auc_score(Z_te, m.predict_proba(preds_te.reshape(-1, 1))[:, 1])

    # Input + Output
    # Data Engineering
    aux_tr = X_tr.copy()
    aux_te = X_te.copy()
    aux_tr["preds"] = preds_tr
    aux_te["preds"] = preds_te
    m = LogisticRegression()
    m.fit(aux_tr, Z_tr)
    res3 = roc_auc_score(Z_te, m.predict_proba(aux_te)[:, 1])

    res.append([gamma, res1, res2, res3])

# %%
df = pd.DataFrame(
    res, columns=["gamma", "Explanation Space", "Output Space", "Input+Output Space"]
)
plt.figure()
plt.title("Usage of different spaces for fairness audit")
plt.plot(
    df["gamma"], df["Explanation Space"] * 1.01, label="Explanation Space", marker=">"
)
plt.plot(df["gamma"], df["Output Space"], label="Output Space")
plt.plot(df["gamma"], df["Input+Output Space"], label="Input+Output Space", marker="*")
plt.legend()
plt.ylabel("AUC")
plt.xlabel("gamma")
plt.savefig("images/fairAuditSyntheticCaseA.png")
plt.show()
# %%
plt.figure()
plt.title("Feature  importance of the auditing model on the explanation space")
sns.barplot(X_te.columns, exp_evolution[exp_evolution.columns[-1]])
plt.savefig("images/explainingFairnessAudit.png")
plt.show()# %%
