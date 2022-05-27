# %%
import pandas as pd
import numpy as np
from synthetic.fair_domain_adaptation_utils import gen_synth_shift_data, random_logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import shap
from sklearn.model_selection import train_test_split
from fairtools.detector import shap_detector
import seaborn as sns

# %%
N = 10_000
gamma = 0
x11 = np.random.normal(0, 1, size=N)
x12 = np.random.normal(0, 1, size=N)
y1 = np.where(x11 + x12 + np.random.normal(0, 2) > 0, 1, 0)
a1 = np.repeat(-1, N)
x21 = np.random.normal(gamma, 1, size=N)
x22 = np.random.normal(gamma, 1, size=N)
a2 = np.repeat(1, N)
y2 = np.where(x21 + x22 + np.random.normal(0, 0.2) > 0, 1, 0)
# %%
X1 = np.concatenate((x11, x21), axis=0)
X2 = np.concatenate((x12, x22), axis=0)
A = np.concatenate((a1, a2), axis=0)
y = np.concatenate((y1, y2), axis=0)
# %%
X = pd.DataFrame([A, X1, X2]).T
X.columns = ["var%d" % (i + 1) for i in range(X.shape[1])]
X["target"] = y
# %%
X = X.sample(frac=1, replace=False)
# %%
X_tr, X_te, y_tr, y_te = train_test_split(
    X.drop(columns=["target"]),
    X.target,
    test_size=0.33,
    random_state=42,
)
# %%
att_tr = X_tr["var1"].values
att_te = X_te["var1"].values
X_tr = X_tr.drop(columns=["var1"])
X_te = X_te.drop(columns=["var1"])
# %%
# model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, verbosity=0,use_label_encoder=False)
model = LogisticRegression()
model.fit(X_tr.values, y_tr.values)
preds_tr = model.predict(X_tr.values)
preds_te = model.predict(X_te.values)
# %%
print("AUC Train:", roc_auc_score(y_tr, model.predict_proba(X_tr.values)[:, 1]))
print("AUC Test:", roc_auc_score(y_te, model.predict_proba(X_te.values)[:, 1]))

# %%
white_tpr = np.mean(preds_tr[(y_tr == 1) & (att_tr == -1)])
black_tpr = np.mean(preds_tr[(y_tr == 1) & (att_tr == 1)])
print("EOF Train: ", white_tpr - black_tpr)
white_tpr = np.mean(preds_te[(y_te == 1) & (att_te == -1)])
black_tpr = np.mean(preds_te[(y_te == 1) & (att_te == 1)])
print("EOF Test: ", white_tpr - black_tpr)
# %%
explainer = shap.Explainer(model)
## Train Data
shapX1 = explainer(X_tr[att_tr == 1]).values
shapX2 = explainer(X_tr[att_tr == -1]).values
print(shap_detector(data1=shapX1, data2=shapX2))
## Test data
shapX1 = explainer(X_te[att_te == 1]).values
shapX2 = explainer(X_te[att_te == -1]).values
print(shap_detector(data1=shapX1, data2=shapX2))


# %%
