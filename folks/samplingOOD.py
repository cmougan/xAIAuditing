# %%
from folktables import ACSDataSource, ACSIncome
import pandas as pd
from collections import defaultdict
from xgboost import XGBRegressor, XGBClassifier
from scipy.stats import kstest, wasserstein_distance
import seaborn as sns

import shap
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.dummy import DummyRegressor

import sys
from tqdm import tqdm

sys.path.append("../")
from fairtools.xaiUtils import ShapEstimator
import random

random.seed(0)
# %%
# Load data
data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
ca_data = data_source.get_data(states=["CA"], download=True)
mi_data = data_source.get_data(states=["MI"], download=True)
tx_data = data_source.get_data(states=["TX"], download=True)

ca_features, ca_labels, ca_group = ACSIncome.df_to_numpy(ca_data)
mi_features, mi_labels, mi_group = ACSIncome.df_to_numpy(mi_data)
tx_features, tx_labels, tx_group = ACSIncome.df_to_numpy(tx_data)
## Conver to DF
ca_features = pd.DataFrame(ca_features, columns=ACSIncome.features)
mi_features = pd.DataFrame(mi_features, columns=ACSIncome.features)
tx_features = pd.DataFrame(tx_features, columns=ACSIncome.features)
# %%
# Modeling
model = XGBClassifier()

# Train on CA data
# preds_ca = cross_val_predict(model, ca_features, ca_labels, cv=3)
model.fit(ca_features, ca_labels)
# %%
# Lets add the target to ease the sampling
mi_full = mi_features.copy()
mi_full["target"] = mi_labels
# %%
################################
####### PARAMETERS #############
SAMPLE_FRAC = 100
ITERS = 500
# %%
# Input KS
train = []
performance = []
train_shap = []

explainer = shap.Explainer(model)
shap_train = explainer(ca_features)
shap_train = pd.DataFrame(shap_train.values, columns=ca_features.columns)

for i in tqdm(range(0, ITERS), leave=False):
    row = []
    row_shap = []

    # Sampling
    aux = mi_full.sample(n=SAMPLE_FRAC, replace=True)

    # Performance calculation
    preds = model.predict_proba(aux.drop(columns="target"))[:, 1]
    performance.append(roc_auc_score(aux.target.values, preds))

    # Shap values calculation
    shap_values = explainer(aux.drop(columns="target"))
    shap_values = pd.DataFrame(shap_values.values, columns=ca_features.columns)

    for feat in ca_features.columns:
        ks = wasserstein_distance(ca_features[feat], aux[feat])
        sh = wasserstein_distance(shap_train[feat], shap_values[feat])
        row.append(ks)
        row_shap.append(sh)

    train_shap.append(row_shap)
    train.append(row)


# Save results
train_df = pd.DataFrame(train)
train_df.columns = ca_features.columns

train_shap_df = pd.DataFrame(train_shap)
train_shap_df.columns = ca_features.columns
train_shap_df = train_shap_df.add_suffix("_shap")

# %%
## ONLY DATA
print("ONLY DATA")
X_train, X_test, y_train, y_test = train_test_split(
    train_df, performance, test_size=0.33, random_state=42
)

modelOOD = DummyRegressor()
modelOOD.fit(X_train, y_train)
print("Dummy")
print(mean_absolute_error(modelOOD.predict(X_test), y_test))

modelOOD = Lasso()
modelOOD.fit(X_train, y_train)
print("Lasso")
print(mean_absolute_error(modelOOD.predict(X_test), y_test))

modelOOD = LinearRegression()
modelOOD.fit(X_train, y_train)
print("Linear Regression")
print(mean_absolute_error(modelOOD.predict(X_test), y_test))

print("Random Forest")
modelOOD = RandomForestRegressor()
modelOOD.fit(X_train, y_train)
print(mean_absolute_error(modelOOD.predict(X_test), y_test))
# %%
#### ONLY SHAP
print("ONLY SHAP")
X_train, X_test, y_train, y_test = train_test_split(
    train_shap_df, performance, test_size=0.33, random_state=42
)

modelOOD = DummyRegressor()
modelOOD.fit(X_train, y_train)
print("Dummy")
print(mean_absolute_error(modelOOD.predict(X_test), y_test))

modelOOD = Lasso()
modelOOD.fit(X_train, y_train)
print("Lasso")
print(mean_absolute_error(modelOOD.predict(X_test), y_test))

modelOOD = LinearRegression()
modelOOD.fit(X_train, y_train)
print("Linear Regression")
print(mean_absolute_error(modelOOD.predict(X_test), y_test))

print("Random Forest")
modelOOD = RandomForestRegressor()
modelOOD.fit(X_train, y_train)
print(mean_absolute_error(modelOOD.predict(X_test), y_test))
# %%
### SHAP + DATA
print("SHAP + DATA")
X_train, X_test, y_train, y_test = train_test_split(
    pd.concat([train_shap_df, train_df], axis=1),
    performance,
    test_size=0.33,
    random_state=42,
)

modelOOD = DummyRegressor()
modelOOD.fit(X_train, y_train)
print("Dummy")
print(mean_absolute_error(modelOOD.predict(X_test), y_test))

modelOOD = Lasso()
modelOOD.fit(X_train, y_train)
print("Lasso")
print(mean_absolute_error(modelOOD.predict(X_test), y_test))

modelOOD = LinearRegression()
modelOOD.fit(X_train, y_train)
print("Linear Regression")
print(mean_absolute_error(modelOOD.predict(X_test), y_test))

print("Random Forest")
modelOOD = RandomForestRegressor()
modelOOD.fit(X_train, y_train)
print(mean_absolute_error(modelOOD.predict(X_test), y_test))
