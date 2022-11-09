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
X, y = data.get_state(year="2014", state="CA")
X_ = X.drop(["group"], axis=1)
# %%
# Train on CA data
coefs = []
aucs = []
for i in tqdm(range(100)):
    # Bootstrap
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.8, random_state=i)
    # Random assign
    X_train["group"] = np.random.randint(0, 2, X_train.shape[0])

    # Train model
    audit = ExplanationAudit(
        model=XGBRegressor(),
        gmodel=Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(penalty="l1", solver="liblinear")),
            ]
        ),
    )
    audit.fit(X_train, y_train, Z="group")

    # Save results
    coefs.append(audit.get_coefs()[0])
    aucs.append(audit.get_auc_val())

# %%
## OOD AUC
ood_auc = []
ood_coefs = []
# states = ["NY14", "TX14", "HI14", "NY18", "TX18", "HI18", "CA18", "CA14"]
states = ["NY18", "TX18", "HI18", "CA18"]

for state in states:
    (
        X,
        y,
    ) = data.get_state(state=state[:2], year="20" + state[2:])
    audit.fit(X, y, Z="group")
    ood_auc.append(audit.get_auc_val())
    ood_coefs.append(audit.get_coefs()[0])


# %%
# Plot AUC
plt.figure(figsize=(10, 6))
plt.title("AUC OOD performance of the Explanation Shift detector")
plt.ylabel("AUC")
sns.kdeplot(aucs, fill=True, label="Randomly assigned groups")
plt.axvline(ood_auc[0], label=states[0], color="#00BFFF")
plt.axvline(ood_auc[1], label=states[1], color="#C68E17")
plt.axvline(ood_auc[2], label=states[2], color="#7DFDFE")
plt.axvline(ood_auc[3], label=states[3], color="#6F4E37")

plt.legend()
plt.tight_layout()
plt.show()

# %%
# Analysis of coeficients
coefs = pd.DataFrame(ood_coefs, columns=X_.columns)
coefs_res = pd.DataFrame(index=coefs.columns)
for i in range(len(ood_coefs)):
    coefs_res[states[i]] = np.mean(coefs <= ood_coefs[i])

# %%
coefs_res["mean"] = coefs_res.mean(axis=1)
coefs_res.sort_values(by="mean", ascending=True)
# %%
coefs_res.sort_values(by="mean", ascending=True).shape
# %%
plt.figure(figsize=(10, 6))
plt.title("Feature importance of the Explanation Shift detector (p-values)")
sns.heatmap(coefs_res.sort_values(by="mean", ascending=True), annot=True)
plt.tight_layout()
plt.savefig("images/feature_importance.png")
plt.show()
# %%
# Global
res = pd.DataFrame(
    audit.get_coefs()[0], columns=["coef"], index=X_.columns
).sort_values("coef", ascending=False)
plt.figure()
plt.title("Global Feature Importance")
plt.ylabel("Absolute value of regression coefficient")
sns.barplot(x=res.index, y=res.coef.abs())
plt.xticks(rotation=45)
plt.savefig("images/folksglobal.png")
plt.show()
# Local
plt.figure()
plt.title("Local Feature Importance")
plt.ylabel("Absolute value of regression coefficient")
ind_coef = np.abs(X_.iloc[40].values * audit.get_coefs()[0])
sns.barplot(x=X_.columns, y=ind_coef)
plt.xticks(rotation=45)
plt.savefig("images/folkslocal.png")
plt.show()
