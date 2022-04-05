# %%
from folktables import ACSDataSource, ACSIncome
import pandas as pd
from collections import defaultdict
from scipy.stats import kstest, wasserstein_distance
import seaborn as sns
import numpy as np
import random

# Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.dummy import DummyRegressor
from sklearn.svm import SVR

# Specific packages
from xgboost import XGBRFClassifier, XGBRegressor, XGBClassifier
import shap
from tqdm import tqdm

# Home made code
import sys

sys.path.append("../")
from fairtools.utils import psi, loop_estimators

# Seeding
np.random.seed(0)
random.seed(0)
# %%
# Load data
data_source = ACSDataSource(survey_year="2014", horizon="1-Year", survey="person")
ca_data = data_source.get_data(states=["CA"], download=True)
data_source = ACSDataSource(survey_year="2016", horizon="1-Year", survey="person")
mi_data = data_source.get_data(states=["MI"], download=True)
data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
tx_data = data_source.get_data(states=["HI"], download=True)

ca_features, ca_labels, ca_group = ACSIncome.df_to_numpy(ca_data)
mi_features, mi_labels, mi_group = ACSIncome.df_to_numpy(mi_data)
tx_features, tx_labels, tx_group = ACSIncome.df_to_numpy(tx_data)
## Conver to DF
ca_features = pd.DataFrame(ca_features, columns=ACSIncome.features)
mi_features = pd.DataFrame(mi_features, columns=ACSIncome.features)
tx_features = pd.DataFrame(tx_features, columns=ACSIncome.features)
# %%
# Modeling
model = XGBClassifier(verbosity=0, silent=True, njobs=1)

# Train on CA data
preds_ca = cross_val_predict(
    model, ca_features, ca_labels, cv=3, method="predict_proba"
)[:, 1]
model.fit(ca_features, ca_labels)
# Test on MI data
preds_mi = model.predict_proba(mi_features)[:, 1]
# Test on TX data
preds_tx = model.predict_proba(tx_features)[:, 1]

##Fairness
white_tpr = np.mean(preds_ca[(ca_labels == 1) & (ca_group == 1)])
black_tpr = np.mean(preds_ca[(ca_labels == 1) & (ca_group == 2)])
eof_tr = white_tpr - black_tpr
print("Train EO", eof_tr)

white_tpr = np.mean(preds_mi[(mi_labels == 1) & (mi_group == 1)])
black_tpr = np.mean(preds_mi[(mi_labels == 1) & (mi_group == 2)])
print("Test MI EO", white_tpr - black_tpr)


white_tpr = np.mean(preds_tx[(tx_labels == 1) & (tx_group == 1)])
black_tpr = np.mean(preds_tx[(tx_labels == 1) & (tx_group == 2)])
print("Test TX EO", white_tpr - black_tpr)

## Model performance
print("CA", roc_auc_score(ca_labels, preds_ca))
print("MI", roc_auc_score(mi_labels, preds_mi))
print("TX", roc_auc_score(tx_labels, preds_tx))
# %%
## Can we learn xAI help to solve this issue?
################################
####### PARAMETERS #############
SAMPLE_FRAC = 1_000
ITERS = 2_0
# Init
train = defaultdict()
train_ood = defaultdict()
performance = defaultdict()
performance_ood = defaultdict()
train_shap = defaultdict()
train_shap_ood = defaultdict()
eof = defaultdict()
eof_ood = defaultdict()

# xAI Train
explainer = shap.Explainer(model)
shap_test = explainer(ca_features)
shap_test = pd.DataFrame(shap_test.values, columns=ca_features.columns)

# Lets add the target to ease the sampling
mi_full = mi_features.copy()
mi_full["group"] = mi_group
mi_full["target"] = mi_labels

# Lets add the target to ease the sampling
tx_full = tx_features.copy()
tx_full["group"] = tx_group
tx_full["target"] = tx_labels

train_error = roc_auc_score(ca_labels, preds_ca)

# %%
# Loop to creat training data
for i in tqdm(range(0, ITERS), leave=False):
    row = []
    row_shap = []
    row_ood = []
    row_shap_ood = []

    # Sampling
    aux = mi_full.sample(n=SAMPLE_FRAC, replace=True)
    aux_ood = tx_full.sample(n=SAMPLE_FRAC, replace=True)

    # Performance calculation
    preds = model.predict_proba(aux.drop(columns=["target", "group"]))[:, 1]
    preds = train_error - preds  # How much the preds differ from train
    performance[i] = mean_absolute_error(aux.target.values, preds)
    ## Fairness
    white_tpr = np.mean(preds[(aux.target == 1) & (aux.group == 1)])
    black_tpr = np.mean(preds[(aux.target == 1) & (aux.group == 2)])
    eof[i] = eof_tr - (white_tpr - black_tpr)

    # OOD performance calculation
    preds_ood = model.predict_proba(aux_ood.drop(columns=["target", "group"]))[:, 1]
    preds_ood = train_error - preds_ood  # How much the preds differ from train
    performance_ood[i] = mean_absolute_error(aux_ood.target.values, preds_ood)
    ## Fairness
    white_tpr = np.mean(preds_ood[(aux_ood.target == 1) & (aux_ood.group == 1)])
    black_tpr = np.mean(preds_ood[(aux_ood.target == 1) & (aux_ood.group == 2)])
    eof_ood[i] = eof_tr - (white_tpr - black_tpr)

    # Shap values calculation
    shap_values = explainer(aux.drop(columns=["target", "group"]))
    shap_values = pd.DataFrame(shap_values.values, columns=ca_features.columns)

    # Shap values calculation OOD
    shap_values_ood = explainer(aux_ood.drop(columns=["target", "group"]))
    shap_values_ood = pd.DataFrame(shap_values_ood.values, columns=ca_features.columns)

    for feat in ca_features.columns:
        # Michigan
        ks = wasserstein_distance(ca_features[feat], aux[feat])
        sh = wasserstein_distance(shap_test[feat], shap_values[feat])
        row.append(ks)
        row_shap.append(sh)

        # OOD
        ks_ood = wasserstein_distance(ca_features[feat], aux_ood[feat])
        sh_ood = wasserstein_distance(shap_test[feat], shap_values_ood[feat])
        row_ood.append(ks_ood)
        row_shap_ood.append(sh_ood)

    # Save test
    train_shap[i] = row_shap
    train[i] = row
    # Save OOD
    train_shap_ood[i] = row_shap_ood
    train_ood[i] = row_ood

# %%
# Save results
## Train (previous test)
train_df = pd.DataFrame(train).T
train_df.columns = ca_features.columns

train_shap_df = pd.DataFrame(train_shap).T
train_shap_df.columns = ca_features.columns
train_shap_df = train_shap_df.add_suffix("_shap")
## Test (previous OOD)
train_df_ood = pd.DataFrame(train_ood).T
train_df_ood.columns = ca_features.columns

train_shap_df_ood = pd.DataFrame(train_shap_ood).T
train_shap_df_ood.columns = ca_features.columns
train_shap_df_ood = train_shap_df_ood.add_suffix("_shap")
## Fairness

# %%
# Estimators for the loop
estimators = defaultdict()
estimators["Dummy"] = DummyRegressor()
estimators["Linear"] = Pipeline(
    [("scaler", StandardScaler()), ("model", LinearRegression())]
)
estimators["RandomForest"] = RandomForestRegressor(random_state=0)
estimators["XGBoost"] = XGBRegressor(
    verbosity=0, verbose=0, silent=True, random_state=0
)
estimators["SVM"] = SVR()
estimators["MLP"] = MLPRegressor(random_state=0)
# %%

## Loop over different G estimators
print("-----------PERFORMANCE-----------")
loop_estimators(
    estimator_set=estimators,
    normal_data=train_df,
    shap_data=train_shap_df,
    normal_data_ood=train_df_ood,
    shap_data_ood=train_shap_df_ood,
    performance_ood=performance_ood,
    target=performance,
)

print("-----------FAIRNESS-----------")
loop_estimators(
    estimator_set=estimators,
    normal_data=train_df,
    shap_data=train_shap_df,
    normal_data_ood=train_df_ood,
    shap_data_ood=train_shap_df_ood,
    performance_ood=eof_ood,
    target=eof,
)

# %%
