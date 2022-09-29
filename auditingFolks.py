# %%
import warnings

warnings.filterwarnings("ignore")

from folktables import (
    ACSDataSource,
    ACSIncome,
    ACSEmployment,
    ACSMobility,
    ACSPublicCoverage,
    ACSTravelTime,
)
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
import numpy as np
import random
import matplotlib.pyplot as plt

# Scikit-Learn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
)

# Specific packages
from xgboost import XGBClassifier
import shap

# Seeding
np.random.seed(0)
random.seed(0)
# %%
# Load data
data_source = ACSDataSource(survey_year="2014", horizon="1-Year", survey="person")
ca_data = data_source.get_data(states=["CA"], download=True)
ca_features, ca_labels, ca_group = ACSIncome.df_to_numpy(ca_data)

## Scale & Conver to DF
ca_features = StandardScaler().fit_transform(ca_features)
ca_features = pd.DataFrame(ca_features, columns=ACSIncome.features)

# Filter to only have groups 1 and 2
ca_features["group"] = ca_group
ca_features["label"] = ca_labels
ca_features = ca_features[(ca_features["group"] == 1) | (ca_features["group"] == 2)]
ca_features["group"] = ca_features["group"].values - 1  # This is to make it 0 and 1
# %%
## Train Test Split
N = ca_features.shape[0]
ca_features = ca_features.sample(frac=1, replace=False, random_state=0).reset_index(
    drop=True
)
train = ca_features.iloc[: int(N / 3)]
val = ca_features.iloc[int(N / 3) : 2 * int(N / 3)]
test = ca_features.iloc[2 * int(N / 3) :]

X_tr = train.drop(["label", "group", "RAC1P"], axis=1)
y_tr = train["label"].values
Z_tr = train["group"].values
X_val = val.drop(["label", "group", "RAC1P"], axis=1)
y_val = val["label"].values
Z_val = val["group"].values
X_te = test.drop(["label", "group", "RAC1P"], axis=1)
y_te = test["label"].values
Z_te = test["group"].values
del ca_features, ca_labels, ca_group
# %%
## Decide the model type of F & G
F = "XGBoost"
G = "XGBoost"
# %%
# Modeling
if F == "Linear":
    model = LogisticRegression()
elif F == "XGBoost":
    model = XGBClassifier(verbosity=0, silent=True)
else:
    raise ValueError("F must be either 'Linear' or 'XGBoost'")

# Train on CA data
model.fit(X_tr, y_tr)
# %%
# SHAP
if F == "Linear":
    explainer = shap.LinearExplainer(
        model, X_te, feature_dependence="correlation_dependent"
    )
elif F == "XGBoost":
    explainer = shap.TreeExplainer(model)
else:
    raise ValueError("F must be either 'Linear' or 'XGBoost'")
shap_tr = explainer(X_tr).values
shap_tr = pd.DataFrame(shap_tr)
shap_tr.columns = X_tr.columns

shap_val = explainer(X_val).values
shap_val = pd.DataFrame(shap_val)
shap_val.columns = X_val.columns

shap_te = explainer(X_te).values
shap_te = pd.DataFrame(shap_te)
shap_te.columns = X_tr.columns

if G == "Linear":
    g = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(penalty="l1", solver="liblinear")),
        ]
    )
elif G == "XGBoost":
    g = XGBClassifier(verbosity=0, silent=True)
else:
    raise ValueError("G must be Linear or XGBoost")

g.fit(shap_val, Z_val)
# %%
# Plotting of Auditing
if G == "Linear":
    res = pd.DataFrame(
        data={
            "feature": shap_te.columns,
            "coef": np.abs(g.named_steps["model"].coef_[0]),
        }
    )
    res.sort_values(by="coef", ascending=False, inplace=True)
    # Global
    plt.figure()
    plt.title("Global Feature Importance")
    plt.ylabel("Absolute value of regression coefficient")
    sns.barplot(x=res.feature, y=res.coef)
    plt.xticks(rotation=45)
    plt.savefig("images/folksglobal.png")
    plt.show()
    # Local
    plt.figure()
    plt.title("Local Feature Importance")
    plt.ylabel("Absolute value of regression coefficient")
    ind_coef = np.abs(X_te.iloc[40].values * g.named_steps["model"].coef_[0])
    sns.barplot(x=res.feature, y=ind_coef)
    plt.xticks(rotation=45)
    plt.savefig("images/folkslocal.png")
    plt.show()
elif G == "XGBoost":
    explainer = shap.TreeExplainer(g)
    shap_values = explainer(X_te)
    # Global
    plt.figure()
    plt.title("Global Feature Importance")
    shap.plots.bar(shap_values, max_display=20, show=False)
    plt.savefig("images/folksglobal.png")
    plt.show()
    # Local
    plt.figure()
    plt.title("Local Feature Importance")
    # visualize the first prediction's explanation
    shap.plots.waterfall(shap_values[0], show=False)
    plt.savefig("images/folkslocal.png")
    plt.show()


# %%
print("Performance of model F")
print(
    "Train: ",
    np.round(roc_auc_score(y_tr, model.predict_proba(X_tr)[:, 1]), decimals=2),
)
print(
    "Val: ",
    np.round(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]), decimals=2),
)
print(
    "Test: ", np.round(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]), decimals=2)
)
# %%
print("Performance of Audit detector G")
print(
    "Train: ", np.round(roc_auc_score(Z_tr, g.predict_proba(shap_tr)[:, 1]), decimals=2)
)
print(
    "Val: ", np.round(roc_auc_score(Z_val, g.predict_proba(shap_val)[:, 1]), decimals=2)
)
print(
    "Test: ", np.round(roc_auc_score(Z_te, g.predict_proba(shap_te)[:, 1]), decimals=2)
)
# %%
