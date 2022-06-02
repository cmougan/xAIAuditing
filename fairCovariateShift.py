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
res = []
for gamma in np.linspace(0, 1, 10):
    x1 = np.random.normal(2, 1, size=N)
    x2 = np.random.normal(4, 1, size=N)
    A = np.random.choice([-1, 1], N)
    x3 = []
    viz1 = []
    viz2 = []
    for i, element in enumerate(A):
        if element > 0:
            val = gamma * x1[i] + 2 * gamma * x2[i] + np.random.normal(0, 0.1)
            x3.append(val)
            viz1.append(val)
        else:
            val = -(gamma * x1[i] + 2 * gamma * x2[i]) + np.random.normal(0, 0.1)
            x3.append(val)
            viz2.append(val)
    # sns.kdeplot(viz1)
    # sns.kdeplot(viz2)

    # y = random_logit(gamma*A * x1 + x2 + np.random.normal(0, 0.02, size=N))
    # y = np.where(y==-1,0,1)
    y = gamma * A + x1 + x2 + np.random.normal(0, 0.02, size=N)

    X = pd.DataFrame([A, x1, x2, x3]).T
    X.columns = ["var%d" % (i + 1) for i in range(X.shape[1])]
    X["target"] = y

    X_tr, X_te, y_tr, y_te = train_test_split(
        X.drop(columns=["target"]),
        X.target,
        test_size=0.33,
        random_state=42,
    )

    att_tr = X_tr["var1"].values
    att_te = X_te["var1"].values
    att_tr_ = np.where(att_tr == -1, 0, 1)
    att_te_ = np.where(att_te == -1, 0, 1)

    X_tr = X_tr.drop(columns=["var1"])
    X_te = X_te.drop(columns=["var1"])

    model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        verbosity=0,
        use_label_encoder=False,
    )
    # model = LogisticRegression()
    model.fit(X_tr.values, y_tr.values)
    preds_tr = model.predict(X_tr.values)
    preds_te = model.predict(X_te.values)
    # preds_tr_ = model.predict_proba(X_tr.values)[:,1]
    # preds_te_ = model.predict_proba(X_te.values)[:,1]

    # print("AUC Train:", roc_auc_score(y_tr, model.predict_proba(X_tr.values)[:, 1]))
    # print("AUC Test:", roc_auc_score(y_te, model.predict_proba(X_te.values)[:, 1]))
    # print("AUC Train:", r2_score(y_tr, model.predict(X_tr.values)))
    # print("AUC Test:", r2_score(y_te, model.predict(X_te.values)))

    white_tpr = np.mean(preds_tr[(y_tr == 1) & (att_tr == -1)])
    black_tpr = np.mean(preds_tr[(y_tr == 1) & (att_tr == 1)])
    # print("EOF Train: ", white_tpr - black_tpr)
    white_tpr = np.mean(preds_te[(y_te == 1) & (att_te == -1)])
    black_tpr = np.mean(preds_te[(y_te == 1) & (att_te == 1)])
    # print("EOF Test: ", white_tpr - black_tpr)

    explainer = shap.Explainer(model)
    ## Train Data
    # shapX1 = explainer(X_tr[att_tr == 1]).values
    # shapX2 = explainer(X_tr[att_tr == -1]).values
    # print(shap_detector(data1=shapX1, data2=shapX2))
    ## Test data
    shapX1 = explainer(X_te[att_te == 1]).values
    shapX2 = explainer(X_te[att_te == -1]).values
    res1 = shap_detector(data1=shapX1, data2=shapX2)

    m = LogisticRegression()
    m.fit(preds_tr.reshape(-1, 1), att_tr_)
    res2 = roc_auc_score(att_te, m.predict_proba(preds_te.reshape(-1, 1))[:, 1])

    # Data Engineering
    aux_tr = X_tr.copy()
    aux_te = X_te.copy()
    aux_tr["preds"] = preds_tr
    aux_te["preds"] = preds_te
    m = LogisticRegression()
    m.fit(aux_tr, att_tr_)
    res3 = roc_auc_score(att_te, m.predict_proba(aux_te)[:, 1])

    res.append([gamma, res1, res2, res3])

# %%
plt.figure()
pd.DataFrame(
    res, columns=["gamma", "Explanation Space", "Output Space", "Input+Output Space"]
).plot(x="gamma", y=["Explanation Space", "Output Space", "Input+Output Space"])
plt.ylabel('AUC')
plt.show()

# %%
