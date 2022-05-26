# %%
import pandas as pd
import numpy as np
from synthetic.fair_domain_adaptation_utils import gen_synth_shift_data, random_logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import shap
from sklearn.model_selection import train_test_split
from fairtools.detector import shap_detector

# %%
data_src, data_tar, sensible_feature, non_separating_feature = gen_synth_shift_data(
    gamma_shift_src=0,
    gamma_shift_tar=0,
    gamma_A=0.0,
    C_src=0,
    C_tar=0,
    N=1000,
    verbose=False,
)
# %%
X_tr = pd.DataFrame(data_src[0][1])
y_tr = data_src[0][2]
X_te = pd.DataFrame(data_tar[0][1])
y_te = data_tar[0][2]
# Convert to DF
X_tr.columns = ["var%d" % (i + 1) for i in range(X_tr.shape[1])]
X_te.columns = ["var%d" % (i + 1) for i in range(X_te.shape[1])]
# Drop the protected attribute
att_tr = X_tr[["var1"]]
att_te = X_te[["var1"]]
# X_tr = X_tr.drop(columns='var1')
# X_te = X_te.drop(columns='var1')
# %%
X_tr["var1"] = np.random.choice([-1, 1], 1000)
# %%
model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, verbosity=0)
model.fit(X_tr, y_tr)
preds_tr = model.predict(X_tr)
preds_te = model.predict(X_te)
# %%
print("AUC Train:", roc_auc_score(y_tr, model.predict_proba(X_tr)[:, 1]))
print("AUC Test:", roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))

# %%
white_tpr = np.mean(preds_tr[(y_tr == 1) & (X_tr.var1 == -1)])
black_tpr = np.mean(preds_tr[(y_tr == 1) & (X_tr.var1 == 1)])
# white_tpr = np.mean(preds_tr[(y_tr == 1) & (att_tr.var1 == -1)])
# black_tpr = np.mean(preds_tr[(y_tr == 1) & (att_tr.var1 == 1)])
print("EOF Train: ", white_tpr - black_tpr)
white_tpr = np.mean(preds_te[(y_te == 1) & (att_te.var1 == -1)])
black_tpr = np.mean(preds_te[(y_te == 1) & (att_te.var1 == 1)])
print("EOF Test: ", white_tpr - black_tpr)
# %%
explainer = shap.Explainer(model)
## Train Data
shapX1 = explainer(X_tr[X_tr.var1 == 1]).values[:, 1:]
shapX2 = explainer(X_tr[X_tr.var1 == -1]).values[:, 1:]
print(shap_detector(data1=shapX1, data2=shapX2))
## Test data
shapX1 = explainer(X_te[att_te.var1 == 1]).values[:, 1:]
shapX2 = explainer(X_te[att_te.var1 == -1]).values[:, 1:]
print(shap_detector(data1=shapX1, data2=shapX2))

# %%
