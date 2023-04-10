# %%
import warnings
from fairtools.detector import ExplanationAudit
from fairtools.datasets import GetData
from tqdm import tqdm
import sys
import lightgbm as lgb

warnings.filterwarnings("ignore")

import pandas as pd
import seaborn as sns
from matplotlib.colors import PowerNorm

sns.set_style("whitegrid")
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

# Scikit-Learn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# Specific packages
from xgboost import XGBRegressor, XGBClassifier

# Seeding
np.random.seed(0)
random.seed(0)
# %%

# Load data
state = "CA"
year = "2014"
data = GetData()
try:
    dataset = sys.argv[1]
    X, y = data.get_state(
        year=year, state=state, verbose=True, group1=1, group2=8, datasets=dataset
    )
except Exception as e:
    # Print error
    print("Error:", e)
    print("No dataset specified, using ACSIncome")
    dataset = "ACSIncome"
    X, y = data.get_state(
        year=year, state=state, verbose=True, group1=1, group2=8, datasets=dataset
    )
print("Dataset:", dataset)
X_ = X.drop(["group"], axis=1)
# %%
# XGB
xgb_list = []
xgb_param = np.linspace(1, 10, 10)
for i in xgb_param:
    audit = ExplanationAudit(
        model=XGBRegressor(n_estimators=int(i), max_depth=int(i)),
        gmodel=LogisticRegression(),
    )
    audit.fit(X, y, Z="group")
    xgb_list.append(audit.get_auc_val())

# %%
# Decision Tree
dt_list = []
dt_param = np.linspace(1, 10, 9)
for i in tqdm(dt_param):
    audit = ExplanationAudit(
        model=DecisionTreeRegressor(max_depth=int(i)), gmodel=LogisticRegression()
    )
    audit.fit(X, y, Z="group")
    dt_list.append(audit.get_auc_val())
# %%
# Random Forest
rf_list = []
rf_param = np.linspace(1, 10, 10)
for i in tqdm(rf_param):
    audit = ExplanationAudit(
        model=lgb.LGBMRegressor(
            # boosting_type="rf",
            n_estimators=int(i),
            max_depth=int(i),
        ),
        gmodel=LogisticRegression(),
    )
    audit.fit(X, y, Z="group")
    rf_list.append(audit.get_auc_val())

# %%
# Plot
plt.figure(figsize=(10, 8))
# XGB
plt.plot(xgb_param, xgb_list, label="XGB")
ci = 1.96 * np.std(xgb_list) / np.sqrt(len(xgb_param))
plt.fill_between(xgb_param, (xgb_list - ci), (xgb_list + ci), alpha=0.1)

# DT
plt.plot(dt_param, dt_list, label="Decision Tree")
ci = 1.96 * np.std(dt_list) / np.sqrt(len(dt_param))
plt.fill_between(dt_param, (dt_list - ci), (dt_list + ci), alpha=0.1)

# RF
plt.plot(rf_param, rf_list, label="Random Forest")
ci = 1.96 * np.std(rf_list) / np.sqrt(len(rf_param))
plt.fill_between(rf_param, (rf_list - ci), (rf_list + ci), alpha=0.1)

plt.xlabel("Max Depth/Hyperparameter")
plt.ylabel("ET Inspector AUC")
plt.legend()
plt.title("Log. Reg. as Equal Treatment Detector")
plt.savefig("images/HyperLogReg.pdf", bbox_inches="tight")
plt.show()
# %%
#######################################
########################################
# XGB
xgb_list = []
xgb_param = np.linspace(1, 10, 10)
for i in tqdm(xgb_param):
    audit = ExplanationAudit(
        model=XGBRegressor(n_estimators=int(i), max_depth=int(i)),
        gmodel=XGBClassifier(),
    )
    audit.fit(X, y, Z="group")
    xgb_list.append(audit.get_auc_val())
# %%
# Decision Tree
dt_list = []
dt_param = np.linspace(1, 10, 9)
for i in tqdm(dt_param):
    audit = ExplanationAudit(
        model=DecisionTreeRegressor(max_depth=int(i)), gmodel=XGBClassifier()
    )
    audit.fit(X, y, Z="group")
    dt_list.append(audit.get_auc_val())
# %%
# Random Forest
rf_list = []
rf_param = np.linspace(1, 10, 10)
for i in tqdm(rf_param):
    audit = ExplanationAudit(
        model=lgb.LGBMRegressor(
            # boosting_type="rf",
            n_estimators=int(i),
            max_depth=int(i),
        ),
        gmodel=XGBClassifier(),
    )
    audit.fit(X, y, Z="group")
    rf_list.append(audit.get_auc_val())


# %%
# Plot
plt.figure(figsize=(10, 8))
# XGB
plt.plot(xgb_param, xgb_list, label="XGB")
ci = 1.96 * np.std(xgb_list) / np.sqrt(len(xgb_param))
plt.fill_between(xgb_param, (xgb_list - ci), (xgb_list + ci), alpha=0.1)

# DT
plt.plot(dt_param, dt_list, label="Decision Tree")
ci = 1.96 * np.std(dt_list) / np.sqrt(len(dt_param))
plt.fill_between(dt_param, (dt_list - ci), (dt_list + ci), alpha=0.1)

# RF
plt.plot(rf_param, rf_list, label="Random Forest")
ci = 1.96 * np.std(rf_list) / np.sqrt(len(rf_param))
plt.fill_between(rf_param, (rf_list - ci), (rf_list + ci), alpha=0.1)

plt.xlabel("Max Depth/Hyperparameter")
plt.ylabel("ET Inspector AUC")
plt.legend()
plt.title("XGB as Equal Treatment INSPECTOR")
plt.savefig("images/HyperXGB.pdf", bbox_inches="tight")
plt.show()
# %%

### Loop over all estimators ###
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier

list_estimator = [
    XGBClassifier(),
    LogisticRegression(),
    Lasso(),
    Ridge(),
    RandomForestRegressor(),
    DecisionTreeRegressor(),
    MLPRegressor(),
]
list_detector = [
    XGBClassifier(),
    LogisticRegression(),
    SVC(probability=True),
    RandomForestClassifier(),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    MLPClassifier(),
]

res = pd.DataFrame(index=range(len(list_estimator)), columns=range(len(list_estimator)))
for i, estimator in tqdm(enumerate(list_estimator)):
    for j, gmodel in enumerate(list_detector):
        detector = ExplanationAudit(model=estimator, gmodel=gmodel, masker=True)
        try:
            audit.fit(X, y, Z="group")

            res.at[i, j] = audit.get_auc_val()
        # Catch errors and print
        except Exception as e:
            print(e)
            res.at[i, j] = np.nan
            print("Error")
            print("Estimator: ", estimator.__class__.__name__)
            print("Detector: ", gmodel.__class__.__name__)

# %%
res.index = [estimator.__class__.__name__ for estimator in list_estimator]
res.columns = [estimator.__class__.__name__ for estimator in list_estimator]
# %%
res.dropna().astype(float).round(3).to_csv("results/ExplanationAudit.csv")
