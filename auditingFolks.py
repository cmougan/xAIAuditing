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
from scipy.stats import wasserstein_distance

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
X, y = data.get_state(year="2014", state="CA", verbose=True)
X_ = X.drop(["group"], axis=1)
# %%
# Train on CA data
cofs = []
aucs = []
for i in tqdm(range(10)):
    # Bootstrap
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.632, random_state=i)
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
    cofs.append(audit.gmodel.steps[-1][1].coef_[0])
    aucs.append(audit.get_auc_val())

# %%
## OOD AUC
ood_auc = {}
ood_coefs = {}
pairs = ["18", "12", "19", "68", "62", "69", "82", "89", "29"]
pairs_named = [
    "White-Other",
    "White-Black",
    "White-Mixed",
    "Asian-Other",
    "Asian-Black",
    "Asian-Mixed",
    "Other-Black",
    "Other-Mixed",
    "Black-Mixed",
]

for pair in tqdm(pairs):
    X_, y_ = data.get_state(
        state="CA",
        year="2014",
        group1=int(pair[0]),
        group2=int(pair[1]),
    )
    ood_temp = []
    ood_coefs_temp = pd.DataFrame(columns=X.columns)
    for i in range(10):
        X = X_.sample(frac=0.132, replace=True)
        y = y_[X.index]

        try:
            audit.fit(X, y, Z="group")
            ood_temp.append(audit.get_auc_val())
            ood_coefs_temp = ood_coefs_temp.append(
                pd.DataFrame(
                    audit.gmodel.steps[-1][1].coef_,
                    columns=X.drop(["group"], axis=1).columns,
                )
            )
        except Exception as e:
            print("Error with pair", pair)
            print(e)

    ood_auc[pair] = ood_temp
    ood_coefs[pair] = ood_coefs_temp


# %%
# Plot AUC
colors = [
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
plt.figure(figsize=(10, 6))
plt.title("AUC performance of the Discrimination Auditor")
plt.xlabel("AUC")
sns.kdeplot(aucs, fill=True, label="Randomly assigned groups")
for i, value in enumerate(pairs):
    # plt.axvline(np.mean(ood_auc[value]), label=pairs_named[i], color=colors[i])
    sns.kdeplot(ood_auc[value], label=pairs_named[i], color=colors[i], fill=True)
plt.legend()
plt.tight_layout()
plt.savefig("images/detector_auc.png")
plt.show()

# %%
# Analysis of coeficients
coefs = pd.DataFrame(cofs, columns=X.drop(["group"], axis=1).columns)
coefs_res = pd.DataFrame(index=coefs.columns)
# for i in range(len(ood_coefs)):
#    coefs_res[pairs_named[i]] = np.mean(cofs <= ood_coefs[i], axis=0)
# Strength of the feature importance
for i, pair in enumerate(pairs):
    for col in coefs.columns:
        coefs_res.loc[col, pairs_named[i]] = wasserstein_distance(
            ood_coefs[pair][col], coefs[col]
        )
# %%
coefs_res["mean"] = coefs_res.mean(axis=1)
coefs_res.sort_values(by="mean", ascending=True)
# %%
coefs_res.sort_values(by="mean", ascending=True).shape
# %%
from matplotlib.colors import PowerNorm

plt.figure(figsize=(10, 6))
plt.title("Feature importance of Explanation Audits")
sns.heatmap(
    coefs_res.sort_values(by="mean", ascending=False),
    annot=True,
    norm=PowerNorm(gamma=0.5),
)
plt.tight_layout()
plt.savefig("images/feature_importance.pdf", bbox_inches="tight")
plt.savefig("images/feature_importance.png")
plt.show()
# %%
# Cluster map
plt.figure(figsize=(10, 6))
plt.title("Feature importance of Explanation Audits")
sns.clustermap(
    coefs_res.sort_values(by="mean", ascending=False),
    annot=True,
    norm=PowerNorm(gamma=0.5),
)

plt.tight_layout()
plt.savefig("images/feature_importance_cluster.pdf", bbox_inches="tight")
plt.savefig("images/feature_importance_cluster.png")
plt.show()
# %%
