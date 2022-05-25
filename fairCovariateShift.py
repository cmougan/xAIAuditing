# %%
import pandas as pd
import numpy as np
from synthetic.fair_domain_adaptation_utils import gen_synth_shift_data, random_logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import shap
from sklearn.model_selection import train_test_split

# %%
data_src, data_tar, sensible_feature, non_separating_feature = gen_synth_shift_data(
    gamma_shift_src=0,
    gamma_shift_tar=1,
    gamma_A=1.0,
    C_src=0,
    C_tar=0,
    N=1000,
    verbose=False,
)
# %%
X_tr = data_src[0][1]
y_tr = data_src[0][2]
X_te = data_tar[0][1]
y_te = data_tar[0][2]
# %%
X_tr[:,0] = np.random.choice([-1,1],1000)
# %%
model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
model.fit(X_tr, y_tr)
preds_tr = model.predict(X_tr)
preds_te = model.predict(X_te)
# %%
print("AUC Train:", roc_auc_score(y_tr, model.predict_proba(X_tr)[:, 1]))
print("AUC Test:", roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))

# %%
white_tpr = np.mean(preds_tr[(y_tr == 1) & (X_tr[:, 0] == -1)])
black_tpr = np.mean(preds_tr[(y_tr == 1) & (X_tr[:, 0] == 1)])
print("EOF Train: ", white_tpr - black_tpr)
white_tpr = np.mean(preds_te[(y_te == 1) & (X_te[:, 0] == -1)])
black_tpr = np.mean(preds_te[(y_te == 1) & (X_te[:, 0] == 1)])
print("EOF Test: ", white_tpr - black_tpr)
# %%
explainer = shap.Explainer(model)
shapX1 = explainer(X_te[X_te[:,0]==1]).values
shapX2 = explainer(X_te[X_te[:,0]==-1]).values

# %%
shapX1 = pd.DataFrame(shapX1,columns=['var1','var2','var3'])
shapX1['target'] = 0
shapX2 = pd.DataFrame(shapX2,columns=['var1','var2','var3'])
shapX2['target'] = 1
shapALL = pd.concat([shapX1,shapX2],axis=0)
# %%
X_train, X_test, y_train, y_test = train_test_split(shapALL.drop(columns=['var1','target']), shapALL.target, test_size=0.33, random_state=42)
# %%
det = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
det.fit(X_train, y_train)
preds_tr = det.predict(X_train)
preds_te = det.predict(X_train)
# %%
print("AUC Train:", roc_auc_score(y_train, det.predict_proba(X_train)[:, 1]))
print("AUC Test:", roc_auc_score(y_test, det.predict_proba(X_test)[:, 1]))
# %%
