# %%
import warnings
from explanationspace import ExplanationAudit
from fairtools.datasets import GetData
from tqdm import tqdm
import sys
from scipy.stats import brunnermunzel

warnings.filterwarnings("ignore")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

plt.style.use("seaborn-whitegrid")
from matplotlib import rcParams

rcParams["axes.labelsize"] = 14
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12
rcParams["figure.figsize"] = 16, 8
rcParams.update({"font.size": 22})

import numpy as np
import random

from scipy.stats import wasserstein_distance

# Scikit-Learn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


# Specific packages
from xgboost import XGBRegressor

# Seeding
np.random.seed(0)
random.seed(0)


# %%
def roc_auc_ci(y_true, y_score, positive=1):
    AUC = roc_auc_score(y_true, y_score)
    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
    Q1 = AUC / (2 - AUC)
    Q2 = 2 * AUC**2 / (1 + AUC)
    SE_AUC = np.sqrt(
        (AUC * (1 - AUC) + (N1 - 1) * (Q1 - AUC**2) + (N2 - 1) * (Q2 - AUC**2))
        / (N1 * N2)
    )
    lower = AUC - 1.96 * SE_AUC
    upper = AUC + 1.96 * SE_AUC
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1
    return (lower, upper)


# %%
# Load data
state = "CA"
year = "2014"
N_b = 2
boots_size = 0.132
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
aucs_test = []
for i in tqdm(range(N_b)):
    # Bootstrap
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=boots_size, random_state=i
    )
    # Random assign
    X_train["group"] = np.random.randint(0, 2, X_train.shape[0])
    Z_train = X_train["group"]
    X_train = X_train.drop(["group"], axis=1)
    X_test["group"] = np.random.randint(0, 2, X_test.shape[0])
    Z_test = X_test["group"]
    X_test = X_test.drop(["group"], axis=1)

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
    audit.fit_pipeline(X=X_train, y=y_train, z=Z_train)

    # Save results
    cofs.append(audit.inspector.steps[-1][1].coef_[0])
    aucs.append(roc_auc_score(Z_test, audit.predict_proba(X_test)[:, 1]))

    # Statistical Test Analysis
    gA = X_test[Z_test == 0]
    gB = X_test[Z_test == 1]
    pA = audit.predict_proba(gA)[:, 1]
    pB = audit.predict_proba(gB)[:, 1]
    auc = roc_auc_score(Z_test, audit.predict_proba(X_test)[:, 1])

    low, high = roc_auc_ci(Z_test, audit.predict(X_test)[:, 1])
    aucs_test.append(
        [
            0,
            auc,
            low,
            high,
            brunnermunzel(pA, pB).pvalue,
            brunnermunzel(pA, pB).statistic,
        ]
    )


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
    X_["label"] = y_
    ood_temp = []
    ood_coefs_temp = pd.DataFrame(columns=X.columns)
    for i in range(N_b):
        # Train test split X,Y,Z
        X_train = X_.sample(frac=boots_size, replace=True)
        y_train = X_train["label"]
        Z_train = X_train["group"]
        X_train = X_train.drop(["label", "group"], axis=1)
        X_test = X_.drop(X_train.index)
        y_test = X_test["label"]
        Z_test = X_test["group"]
        X_test = X_test.drop(["label", "group"], axis=1)

        try:
            audit.fit_pipeline(X=X_train, y=y_train, z=Z_train)
            ood_temp.append(roc_auc_score(Z_test, audit.predict_proba(X_test)[:, 1]))
            ood_coefs_temp = ood_coefs_temp.append(
                pd.DataFrame(
                    audit.inspector.steps[-1][1].coef_,
                    columns=X.drop(["group"], axis=1).columns,
                )
            )

        except Exception as e:
            print("Error with pair", pair)
            print(e)

    # Statistical Test Analysis
    gA = X_test[Z_test == 0]
    gB = X_test[Z_test == 1]
    pA = audit.predict_proba(gA)[:, 1]
    pB = audit.predict_proba(gB)[:, 1]
    brunnermunzel(pA, pB)
    auc = roc_auc_score(Z_test, audit.predict_proba(X_test)[:, 1])

    low, high = roc_auc_ci(Z_test, audit.predict(X_test)[:, 1])
    aucs_test.append(
        [
            pair,
            auc,
            low,
            high,
            brunnermunzel(pA, pB).pvalue,
            brunnermunzel(pA, pB).statistic,
        ]
    )

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
plt.title("AUC performance of the Equal Treatment Inspector")
plt.xlabel("AUC")
sns.kdeplot(aucs, fill=True, label="Randomly assigned groups")
for i, value in enumerate(pairs):
    # plt.axvline(np.mean(ood_auc[value]), label=pairs_named[i], color=colors[i])
    sns.kdeplot(ood_auc[value], label=pairs_named[i], color=colors[i], fill=True)
plt.legend()
plt.tight_layout()
plt.savefig("images/detector_auc_{}.pdf".format(dataset), bbox_inches="tight")


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
plt.savefig("images/feature_importance_{}.pdf".format(dataset), bbox_inches="tight")
plt.close()
# %%
auc_test = pd.DataFrame(
    aucs_test, columns=["pair", "auc", "low", "high", "pvalue", "statistic"]
)
# %%
auc_test

# %%

roc_auc_ci(Z_test, audit.predict_proba(X_test)[:, 1])
# %%
