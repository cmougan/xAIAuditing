# %%
import pandas as pd
import numpy as np
from synthetic.fair_domain_adaptation_utils import gen_synth_shift_data, random_logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

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
model = LogisticRegression()
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

# %%
