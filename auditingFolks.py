# %%
import warnings
from fairtools.detector import ExplanationAudit

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
from xgboost import XGBRegressor
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
# Split data
X = ca_features.drop(["label", "RAC1P"], axis=1)
X_ = X.drop(["group"], axis=1)
y = ca_features["group"]

# Train on CA data
audit = ExplanationAudit(model=XGBRegressor(), gmodel=LogisticRegression())
audit.fit(X, y, Z="group")
# %%
print("f: ", audit.get_auc_f_val())
print("g: ", audit.get_auc_val())

# %%
# Global
plt.figure()
plt.title("Global Feature Importance")
plt.ylabel("Absolute value of regression coefficient")
sns.barplot(x=X_.columns, y=audit.get_coefs()[0])
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
