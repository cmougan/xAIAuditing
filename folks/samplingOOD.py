# %%
from folktables import ACSDataSource, ACSIncome
import pandas as pd
from collections import defaultdict
from xgboost import XGBRegressor, XGBClassifier
from scipy.stats import kstest
import shap
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score, mean_squared_error
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
# Lets add the target
mi_full = mi_features.copy()
mi_full["target"] = mi_labels
# %%
################################
####### PARAMETERS #############
SAMPLE_FRAC = 100
ITERS = 1_000
# %%
# Input KS
train = []
performance = []
for i in tqdm(range(0, ITERS), leave=False):
    row = []
    aux = mi_full.sample(n=SAMPLE_FRAC, replace=True)
    preds = model.predict_proba(aux.drop(columns="target"))[:, 1]
    performance.append(roc_auc_score(aux.target.values, preds))
    for feat in ca_features.columns:
        ks = kstest(ca_features[feat], aux[feat]).statistic
        row.append(ks)
    train.append(row)
# Save results
train_df = pd.DataFrame(train)
train_df.columns = ca_features.columns

# %%
X_train, X_test, y_train, y_test = train_test_split(
    train_df, performance, test_size=0.33, random_state=42
)
# %%
# Dummy
modelOOD = DummyRegressor()
modelOOD.fit(X_train, y_train)
print(mean_squared_error(modelOOD.predict(X_test), y_test))
# %%
modelOOD = Lasso()
modelOOD.fit(X_train, y_train)
print(mean_squared_error(modelOOD.predict(X_test), y_test))

# %%
# %%
modelOOD = LinearRegression()
modelOOD.fit(X_train, y_train)
print(mean_squared_error(modelOOD.predict(X_test), y_test))
# %%
