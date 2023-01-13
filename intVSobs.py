# %%
import warnings
from fairtools.detector import ExplanationAudit
from fairtools.datasets import GetData
from tqdm import tqdm

warnings.filterwarnings("ignore")

import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
import numpy as np
import random
import matplotlib.pyplot as plt

# Scikit-Learn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Specific packages
from xgboost import XGBRegressor

# Seeding
np.random.seed(0)
random.seed(0)
# %%
# Load data
data = GetData()
X, y = data.get_state(
    state="CA",
    year="2014",
    group1=int(1),
    group2=int(8),
)
# Preprocess
sc = StandardScaler()
Z = X["group"]
X = X.drop(["group"], axis=1)
X = pd.DataFrame(sc.fit_transform(X), columns=X.columns)
## Dont scale the group
X["group"] = Z.values
# %%
# Interventional
audit_int = ExplanationAudit(
    model=LogisticRegression(penalty="l1", solver="liblinear"),
    gmodel=Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(penalty="l1", solver="liblinear")),
        ]
    ),
    masker=True,
    feature_perturbation="interventional",
    algorithm="linear",
)
audit_int.fit(X, y, Z="group")

# %%
# Observational
audit_obs = ExplanationAudit(
    model=LogisticRegression(penalty="l1", solver="liblinear"),
    gmodel=Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(penalty="l1", solver="liblinear")),
        ]
    ),
    masker=True,
    feature_perturbation="correlation_dependent",
    algorithm="linear",
)
audit_obs.fit(
    X,
    y,
    Z="group",
)

# %%
# Groupedbar plot of observed vs intervention using X.columns
cols = set(X.columns) - set(["group"])
width = 0.35
plt.figure(figsize=(10, 5))
plt.bar(
    np.arange(len(cols)) - width / 2,
    audit_int.gmodel.steps[-1][1].coef_[0],
    label="Interventional",
    alpha=0.5,
)
plt.bar(
    np.arange(len(cols)) + width / 2,
    audit_obs.gmodel.steps[-1][1].coef_[0],
    label="Observational",
    alpha=0.5,
)
plt.xticks(np.arange(len(cols)), cols, rotation=90)
plt.legend()
plt.show()
# %%
s1 = audit_int.get_explanations(X.drop(["group"], axis=1))
s2 = audit_obs.get_explanations(X.drop(["group"], axis=1))
# %%
comp = pd.DataFrame([s1.sum(axis=0), s2.sum(axis=0)]).T
comp.columns = ["Interventional", "Observational"]
comp["diff"] = comp["Interventional"] - comp["Observational"]
comp["diff_rel"] = comp["diff"] / comp["Observational"]
# %%
coefss = pd.DataFrame(
    [audit_obs.gmodel.steps[-1][1].coef_[0], audit_int.gmodel.steps[-1][1].coef_[0]]
).T
coefss.columns = ["Observational", "Interventional"]
coefss["diff"] = coefss["Interventional"] - coefss["Observational"]
coefss["diff_rel"] = coefss["diff"] / coefss["Observational"]
coefss.index = cols
# %%
coefss
# %%
audit_int.get_auc_val()
# %%
audit_obs.get_auc_val()
# %%
