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
sys.path.append("../")
from fairtools.xaiUtils import ShapEstimator
from fairtools.utils import psi

# Seeding
np.random.seed(0)
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
model = XGBClassifier(verbosity=0, silent=True)

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
print("Train EO", white_tpr - black_tpr)

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
SAMPLE_FRAC = 100
ITERS = 2_000
# Init
train = defaultdict()
performance = defaultdict()
train_shap = defaultdict()

# xAI Train
explainer = shap.Explainer(model)
shap_test = explainer(ca_features)
shap_test = pd.DataFrame(shap_test.values, columns=ca_features.columns)

# Lets add the target to ease the sampling
mi_full = mi_features.copy()
mi_full["target"] = mi_labels

train_error = roc_auc_score(ca_labels, preds_ca)

# %%
# Loop to creat training data
for i in tqdm(range(0, ITERS), leave=False):
    row = []
    row_shap = []

    # Sampling
    aux = mi_full.sample(n=SAMPLE_FRAC, replace=True)

    # Performance calculation
    preds = model.predict_proba(aux.drop(columns="target"))[:, 1]
    preds = train_error - preds  # How much the preds differ from train
    performance[i] = mean_absolute_error(aux.target.values, preds)

    # Shap values calculation
    shap_values = explainer(aux.drop(columns="target"))
    shap_values = pd.DataFrame(shap_values.values, columns=ca_features.columns)

    for feat in ca_features.columns:
        ks = wasserstein_distance(ca_features[feat], aux[feat])
        sh = wasserstein_distance(shap_test[feat], shap_values[feat])
        row.append(ks)
        row_shap.append(sh)

    train_shap[i] = row_shap
    train[i] = row

# %%
# Save results
train_df = pd.DataFrame(train).T
train_df.columns = ca_features.columns

train_shap_df = pd.DataFrame(train_shap).T
train_shap_df.columns = ca_features.columns
train_shap_df = train_shap_df.add_suffix("_shap")

# %%
# Estimators for the loop
estimators = defaultdict()
estimators["Dummy"] = DummyRegressor()
estimators["Linear"] = Pipeline(
    [("scaler", StandardScaler()), ("model", LinearRegression())]
)
estimators["Lasso"] = Pipeline([("scaler", StandardScaler()), ("model", Lasso())])
estimators["RandomForest"] = RandomForestRegressor(random_state=0)
estimators["XGBoost"] = XGBRegressor(
    verbosity=0, verbose=0, silent=True, random_state=0
)
estimators["SVM"] = SVR()
estimators["MLP"] = MLPRegressor(random_state=0)
# %%
# Loop over different G estimators
for estimator in estimators:
    print(estimator)
    ## ONLY DATA
    X_train, X_test, y_train, y_test = train_test_split(
        train_df, performance, test_size=0.33, random_state=42
    )
    estimators[estimator].fit(X_train, y_train)
    print(
        "ONLY DATA", mean_absolute_error(estimators[estimator].predict(X_test), y_test)
    )

    #### ONLY SHAP
    X_train, X_test, y_train, y_test = train_test_split(
        train_shap_df, performance, test_size=0.33, random_state=42
    )
    estimators[estimator].fit(X_train, y_train)
    print(
        "ONLY SHAP", mean_absolute_error(estimators[estimator].predict(X_test), y_test)
    )

    ### SHAP + DATA
    X_train, X_test, y_train, y_test = train_test_split(
        pd.concat([train_shap_df, train_df], axis=1),
        performance,
        test_size=0.33,
        random_state=42,
    )
    estimators[estimator].fit(X_train, y_train)
    print(
        "SHAP + DATA",
        mean_absolute_error(estimators[estimator].predict(X_test), y_test),
    )

# %%
