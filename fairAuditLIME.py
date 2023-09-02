# %%
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier, XGBRegressor
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from tqdm import tqdm
import pdb

plt.style.use("seaborn-whitegrid")
from matplotlib import rcParams
import lime.lime_tabular

rcParams["axes.labelsize"] = 14
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12
rcParams["figure.figsize"] = 16, 8
rcParams.update({"font.size": 22})


# %%
def create_explanation(X, model):
    exp = X.copy()[:0]
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X.head(100).values,
        feature_names=X.columns,
        discretize_continuous=True,
        verbose=False,
        mode="regression",
    )

    for i, _ in tqdm(enumerate(X.iterrows())):
        ex = explainer.explain_instance(X.iloc[i], model.predict)
        exx = pd.DataFrame(ex.local_exp[0], columns=["feature", "weight"]).sort_values(
            "feature"
        )
        exx.feature = X.columns
        exx = exx.T
        # Make header first row
        new_header = exx.iloc[0]  # grab the first row for the header
        exx = exx[1:]  # take the data less the header row
        exx.columns = new_header
        exx.reset_index(inplace=True)
        exp = pd.concat([exp, exx])

    return exp


# %%
def train_esd(X, Z, model, detector):
    """
    LIME explanation distribution Detector
    """
    aux = create_explanation(X, model)
    aux["y"] = Z

    X_tr, X_te, y_tr, y_te = train_test_split(
        aux.drop(["y", "index"], axis=1), aux["y"], test_size=0.5, random_state=42
    )

    detector.fit(X_tr, y_tr)

    # return auc
    return roc_auc_score(y_te, detector.predict_proba(X_te)[:, 1])


# %%
N = 5_000
CASE_A = False
dp = []
res = []
exp_evolution = pd.DataFrame()
for gamma in np.linspace(0, 1, 20):
    x1 = np.random.normal(1, 1, size=N)
    x2 = np.random.normal(1, 1, size=N)
    x34 = np.random.multivariate_normal([1, 1], [[1, gamma], [gamma, 1]], size=N)
    x3 = x34[:, 0]
    x4 = x34[:, 1]
    # Binarize protected attribute
    x4 = np.where(x4 > np.mean(x4), 1, 0)

    X = pd.DataFrame([x1, x2, x3, x4]).T
    X.columns = ["var%d" % (i + 1) for i in range(X.shape[1])]

    # Different exp -- Remember to change the name of the visualization
    # Case A
    if CASE_A:
        y = (x1 + x2 + x3) / 3
    else:
        # Case B
        y = (x1 + x2) / 3

    y = 1 / (1 + np.exp(-y))

    # Train test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=0.33,
        random_state=42,
    )
    Z_tr = X_tr.var4.values
    Z_te = X_te.var4.values

    # Drop protected attribute
    X_tr = X_tr.drop("var4", axis=1)
    X_te = X_te.drop("var4", axis=1)

    # Learning of the model F
    model = LinearRegression()
    model.fit(X_tr, y_tr)

    preds_tr = model.predict(X_tr)
    preds_te = model.predict(X_te)

    # Demographic parity calculation
    dist = wasserstein_distance(
        model.predict(X_te[Z_te == 0]), model.predict(X_te[Z_te == 1])
    )
    dp.append(dist)
    # Visualizations of distribution of predictions
    # plt.figure()
    # sns.kdeplot(model.predict(X_te[Z_te == 0]), label="0", shade=True)
    # sns.kdeplot(model.predict(X_te[Z_te == 1]), label="1", shade=True)
    # plt.show()

    # SHAP
    explainer = shap.LinearExplainer(
        model, X_tr, feature_dependence="correlation_dependent"
    )
    # explainer = shap.Explainer(model)
    shapX1 = explainer(X_tr).values
    shapX1 = pd.DataFrame(shapX1)
    shapX1.columns = ["var%d" % (i + 1) for i in range(shapX1.shape[1])]
    shapX2 = explainer(X_te).values
    shapX2 = pd.DataFrame(shapX2)
    shapX2.columns = ["var%d" % (i + 1) for i in range(shapX2.shape[1])]

    m = LogisticRegression()
    m.fit(shapX1, Z_tr)
    res1 = roc_auc_score(Z_te, m.predict_proba(shapX2)[:, 1])
    exp_evolution[gamma] = pd.DataFrame(m.coef_.squeeze())

    # Output
    m = LogisticRegression()
    m.fit(preds_tr.reshape(-1, 1), Z_tr)
    res2 = roc_auc_score(Z_te, m.predict_proba(preds_te.reshape(-1, 1))[:, 1])

    # Input
    m = LogisticRegression()
    m.fit(X_tr, Z_tr)
    res3 = roc_auc_score(Z_te, m.predict_proba(X_te)[:, 1])

    # Input + Output
    # Data Engineering
    aux_tr = X_tr.copy()
    aux_te = X_te.copy()
    aux_tr["preds"] = preds_tr
    aux_te["preds"] = preds_te
    m = LogisticRegression()
    m.fit(aux_tr, Z_tr)
    res4 = roc_auc_score(Z_te, m.predict_proba(aux_te)[:, 1])

    # Lime
    # Binarize y_tr

    res5 = train_esd(
        X_te,
        Z_te,
        model,
        LogisticRegression(),
    )

    res.append([gamma, res1, res2, res3, res4, res5])
# %%
yy_tr = np.where(y_tr > np.mean(y_tr), 1, 0)
train_esd(
    X_te,
    Z_te,
    XGBClassifier().fit(X_tr, yy_tr),
    LogisticRegression(),
)
# %%

# %%
create_explanation(X_tr, XGBClassifier().fit(X_tr, yy_tr))
# %%
df = pd.DataFrame(
    res,
    columns=[
        "gamma",
        "Explanation Distributions",
        "Output Distributions",
        "Input Distributions",
        "Input+Output Distributions",
        "Lime",
    ],
)
plt.figure()
plt.title("{} Case".format("Indirect" if CASE_A else "Uninformative"))


# Explanation distribution
plt.plot(
    df["gamma"],
    df["Explanation Distributions"] * 1.01,
    label=r"SHAP",
    marker=">",
)

# Lime
plt.plot(
    df["gamma"],
    df["Lime"],
    label=r"LIME",
    marker="x",
)

plt.ylabel("AUC")
plt.xlabel("gamma")
if CASE_A:
    plt.legend()
    plt.savefig("images/fairAuditSyntheticCaseALIME.pdf", bbox_inches="tight")
else:
    plt.savefig("images/fairAuditSyntheticCaseBLIME.pdf", bbox_inches="tight")

plt.show()
# %%
plt.figure()
plt.title(
    "Global feature importance of the auditing model on the explanation distribution"
)
sns.barplot(X_te.columns, exp_evolution[exp_evolution.columns[-1]])
plt.tight_layout()
plt.savefig("images/explainingFairnessAudit.png")
plt.show()
# %%
## Fit again the fairness auditor with the last gamma for viz purposes
m = LogisticRegression()
m.fit(shapX1, Z_tr)
explainer = shap.Explainer(m.predict, shapX1)
shap_values = explainer(shapX1.head(1))
plt.figure()
plt.title(
    "Local feature importance of the auditing model on the explanation distribution"
)
shap.plots.waterfall(shap_values[0], show=False)
plt.tight_layout()
plt.savefig("images/explainingFairnessAuditLocal.png")
plt.show()
# %%
# %%
