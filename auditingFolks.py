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
import pdb

sns.set_style("whitegrid")
import numpy as np
import random
import matplotlib.pyplot as plt

# Scikit-Learn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (
    roc_auc_score,
)

from sklearn.model_selection import train_test_split

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
# %%
# %%
## Scale & Conver to DF
ca_features = StandardScaler().fit_transform(ca_features)
ca_features = pd.DataFrame(ca_features, columns=ACSIncome.features)

# Filter to only have groups 1 and 2
ca_features["group"] = ca_group
ca_features["label"] = ca_labels
ca_features = ca_features[(ca_features["group"] == 1) | (ca_features["group"] == 2)]
ca_labels = ca_features["label"].values
ca_group = ca_features["group"].values
ca_group = ca_group - 1  # This is to make it 0 and 1
ca_features = ca_features.drop(["group", "label", "RAC1P"], axis=1)
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
preds_ca = cross_val_predict(
    model, ca_features, ca_labels, cv=5, method="predict_proba"
)[:, 1]
model.fit(ca_features, ca_labels)
# %%
# SHAP
if F == "Linear":
    explainer = shap.LinearExplainer(
        model, ca_features, feature_dependence="correlation_dependent"
    )
elif F == "XGBoost":
    explainer = shap.TreeExplainer(model)
else:
    raise ValueError("F must be either 'Linear' or 'XGBoost'")

shapX1 = explainer(ca_features).values
shapX1 = pd.DataFrame(shapX1)
shapX1.columns = ca_features.columns

if G == "Linear":
    g = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())])
elif G == "XGBoost":
    g = XGBClassifier(verbosity=0, silent=True)
else:
    raise ValueError("G must be Linear or XGBoost")

g.fit(shapX1, ca_group)
res1 = roc_auc_score(ca_group, g.predict_proba(shapX1)[:, 1])

# %%
# Plotting of Auditing
if G == "Linear":
    res = pd.DataFrame(
        data={"feature": shapX1.columns, "coef": g.named_steps["model"].coef_[0]}
    )
    res.sort_values(by="coef", ascending=False, inplace=True)
    # Global
    plt.figure()
    plt.title("Global Feature Importance")
    sns.barplot(x=res.feature, y=res.coef)
    plt.xticks(rotation=45)
    plt.show()
    # Local
    plt.figure()
    plt.title("Local Feature Importance")
    ind_coef = shapX1.head(444).values * g.named_steps["model"].coef_[0]
    sns.barplot(x=res.feature, y=ind_coef[0])
    plt.xticks(rotation=45)
    plt.show()
elif G == "XGBoost":
    explainer = shap.TreeExplainer(g)
    shap_values = explainer(ca_features)
    # Global
    plt.figure()
    plt.title("Global Feature Importance")
    shap.plots.bar(shap_values, max_display=20, show=False)
    plt.show()
    # Local
    plt.figure()
    plt.title("Local Feature Importance")
    # visualize the first prediction's explanation
    shap.plots.waterfall(shap_values[0], show=False)
    plt.show()


# %%
