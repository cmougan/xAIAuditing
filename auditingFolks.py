# %%
import warnings
from fairtools.detector import ExplanationAudit
from fairtools.datasets import GetData
from tqdm import tqdm
import sys

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


# Specific packages
from xgboost import XGBRegressor

# Seeding
np.random.seed(0)
random.seed(0)
# %%

# Load data
state = "CA"
year = "2014"
N_b = 20
data = GetData()
try:
    dataset = sys.argv[1]
    X, y = data.get_state(year=year, state=state, verbose=True, datasets=dataset)
except Exception as e:
    # Print error
    print("Error:", e)
    print("No dataset specified, using ACSIncome")
    dataset = "ACSIncome"
    X, y = data.get_state(year=year, state=state, verbose=True, datasets=dataset)
print("Dataset:", dataset)
X_ = X.drop(["group"], axis=1)
# %%
# Train on CA data
cofs = []
aucs = []
for i in tqdm(range(N_b)):
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
        state=state,
        year=year,
        group1=int(pair[0]),
        group2=int(pair[1]),
        verbose=True,
        datasets=dataset,
    )
    ood_temp = []
    ood_coefs_temp = pd.DataFrame(columns=X.columns)
    for i in range(N_b):
        X = X_.sample(frac=0.632, replace=True)
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
plt.title("AUC performance of the Demographic Parity Inspector")
plt.xlabel("AUC")
sns.kdeplot(aucs, fill=True, label="Randomly assigned groups")
for i, value in enumerate(pairs):
    # plt.axvline(np.mean(ood_auc[value]), label=pairs_named[i], color=colors[i])
    sns.kdeplot(ood_auc[value], label=pairs_named[i], color=colors[i], fill=True)
plt.legend()
plt.tight_layout()
plt.savefig("images/detector_auc_{}.png".format(dataset))
plt.close()

# %%
# Analysis of coeficients
coefs = pd.DataFrame(cofs, columns=X.drop(["group"], axis=1).columns)
if "State" in coefs.columns:
    coefs = coefs.drop(["State"], axis=1)
if "NATIVITY" in coefs.columns:
    coefs = coefs.drop(["NATIVITY"], axis=1)
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
# Means on both axis
coefs_res["mean"] = coefs_res.mean(axis=1)
coefs_res.loc["mean"] = coefs_res.mean(axis=0)
coefs_res.sort_values(by="mean", ascending=True)
# %%
plt.figure(figsize=(10, 6))
plt.title("Feature importance of Explanation Audits")
sns.heatmap(
    coefs_res.sort_values(by="mean", ascending=False, axis=0)
    .sort_values(by="mean", ascending=False, axis=1)
    .drop(["mean"], axis=1)
    .drop(["mean"], axis=0),
    annot=True,
    norm=PowerNorm(gamma=0.5),
)
plt.tight_layout()
plt.savefig("images/feature_importance_{}.png".format(dataset))
plt.close()
# %%
