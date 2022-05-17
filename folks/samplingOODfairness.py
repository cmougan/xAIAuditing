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
from collections import defaultdict
from scipy.stats import kstest, wasserstein_distance
import seaborn as sns
import numpy as np
import random
import sys

# Scikit-Learn
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
from xgboost import XGBRegressor, XGBClassifier
import shap
from tqdm import tqdm


# Home made code
import sys

sys.path.append("../")
from fairtools.utils import psi, loop_estimators, loop_estimators_fairness

# Seeding
np.random.seed(0)
random.seed(0)

# Load data
data_source = ACSDataSource(survey_year="2014", horizon="1-Year", survey="person")
ca_data = data_source.get_data(states=["CA"], download=True)
data_source = ACSDataSource(survey_year="2016", horizon="1-Year", survey="person")
mi_data = data_source.get_data(states=["MI"], download=True)

states = [
    "TN",
    "CT",
    "OH",
    "NE",
    "IL",
    "FL",
    "OK",
    "PA",
    "KS",
    "IA",
    "KY",
    "NY",
    "LA",
    "TX",
    "UT",
    "OR",
    "ME",
    "NJ",
    "ID",
    "DE",
    "MN",
    "WI",
    "CA",
    "MO",
    "MD",
    "NV",
    "HI",
    "IN",
    "WV",
    "MT",
    "WY",
    "ND",
    "SD",
    "GA",
    "NM",
    "AZ",
    "VA",
    "MA",
    "AA",
    "NC",
    "SC",
    "DC",
    "VT",
    "AR",
    "WA",
    "CO",
    "NH",
    "MS",
    "AK",
    "RI",
    "AL",
    "PR",
    "MI",
]

data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")


ca_features, ca_labels, ca_group = ACSIncome.df_to_numpy(ca_data)
mi_features, mi_labels, mi_group = ACSIncome.df_to_numpy(mi_data)

## Conver to DF
ca_features = pd.DataFrame(ca_features, columns=ACSIncome.features)
mi_features = pd.DataFrame(mi_features, columns=ACSIncome.features)

# Modeling
model = XGBClassifier(verbosity=0, silent=True, use_label_encoder=False, njobs=1)

# Train on CA data
preds_ca = cross_val_predict(
    model, ca_features, ca_labels, cv=3, method="predict_proba"
)[:, 1]
model.fit(ca_features, ca_labels)
# Test on MI data
preds_mi = model.predict_proba(mi_features)[:, 1]


##Fairness
white_tpr = np.mean(preds_ca[(ca_labels == 1) & (ca_group == 1)])
black_tpr = np.mean(preds_ca[(ca_labels == 1) & (ca_group == 2)])
eof_tr = white_tpr - black_tpr
tpr_tr_one = white_tpr
tpr_tr_two = black_tpr
# print("Train EO", eof_tr)

white_tpr = np.mean(preds_mi[(mi_labels == 1) & (mi_group == 1)])
black_tpr = np.mean(preds_mi[(mi_labels == 1) & (mi_group == 2)])
# print("Test MI EO", white_tpr - black_tpr)

## Can we learn to solve this issue?
################################
####### PARAMETERS #############
SAMPLE_FRAC = 1_000
ITERS = 2_000
# Init
train_one = defaultdict()
train_two = defaultdict()
train_ood_one = defaultdict()
train_ood_two = defaultdict()
performance = defaultdict()
performance_ood = defaultdict()
train_shap_one = defaultdict()
train_shap_two = defaultdict()
train_shap_ood_one = defaultdict()
train_shap_ood_two = defaultdict()

train_one_target_shift = defaultdict()
train_two_target_shift = defaultdict()
train_one_target_shift_ood = defaultdict()
train_two_target_shift_ood = defaultdict()
tpr_one = defaultdict()
tpr_two = defaultdict()
tpr_ood_one = defaultdict()
tpr_ood_two = defaultdict()
train_error = roc_auc_score(ca_labels, preds_ca)

# xAI Train
explainer = shap.Explainer(model)
shap_test = explainer(ca_features)
shap_test = pd.DataFrame(shap_test.values, columns=ca_features.columns)

# Lets add the target to ease the sampling
mi_full = mi_features.copy()
mi_full["group"] = mi_group
mi_full["target"] = mi_labels
# %%
# Trainning set
for i in tqdm(range(0, ITERS), leave=False, desc="Test Bootstrap", position=1):
    # Initiate
    row_one = []
    row_two = []
    row_shap_one = []
    row_shap_two = []

    # Sampling
    aux = mi_full.sample(n=SAMPLE_FRAC, replace=True)

    # Performance calculation
    preds = model.predict_proba(aux.drop(columns=["target", "group"]))[:, 1]
    performance[i] = train_error - roc_auc_score(aux.target, preds)
    ## Fairness
    white_tpr = np.mean(preds[(aux.target == 1) & (aux.group == 1)])
    black_tpr = np.mean(preds[(aux.target == 1) & (aux.group == 2)])
    tpr_one[i] = white_tpr
    tpr_two[i] = black_tpr

    # Shap values calculation
    shap_values = explainer(aux.drop(columns=["target", "group"]))
    shap_values = pd.DataFrame(shap_values.values, columns=ca_features.columns)
    shap_values["group"] = aux.group.values

    for feat in ca_features.columns:
        # Michigan
        ks_one = wasserstein_distance(
            ca_features[ca_group == 1][feat], aux[aux["group"] == 1][feat]
        )
        ks_two = wasserstein_distance(
            ca_features[ca_group == 2][feat], aux[aux["group"] == 2][feat]
        )
        sh_one = wasserstein_distance(
            shap_test[ca_group == 1][feat], shap_values[shap_values["group"] == 1][feat]
        )
        sh_two = wasserstein_distance(
            shap_test[ca_group == 2][feat], shap_values[shap_values["group"] == 2][feat]
        )
        row_one.append(ks_one)
        row_two.append(ks_two)
        row_shap_one.append(sh_one)
        row_shap_two.append(sh_two)
    # Target shift
    ks_one_target = wasserstein_distance(
        ca_labels[ca_group == 1], aux[aux["group"] == 1]["target"]
    )
    ks_two_target = wasserstein_distance(
        ca_labels[ca_group == 2], aux[aux["group"] == 2]["target"]
    )

    train_one_target_shift[i] = ks_one_target
    train_two_target_shift[i] = ks_two_target

    # Save test
    train_shap_one[i] = row_shap_one
    train_one[i] = row_one
    train_shap_two[i] = row_shap_two
    train_two[i] = row_two


# Some transformations
## Train (previous test)
train_df_one = pd.DataFrame(train_one).T
train_df_one.columns = ca_features.columns
train_df_one = train_df_one.add_suffix("_one")
train_df_two = pd.DataFrame(train_two).T
train_df_two.columns = ca_features.columns
train_df_two = train_df_one.add_suffix("_two")

train_shap_df_one = pd.DataFrame(train_shap_one).T
train_shap_df_one.columns = ca_features.columns
train_shap_df_one = train_shap_df_one.add_suffix("_shap_one")
train_shap_df_two = pd.DataFrame(train_shap_two).T
train_shap_df_two.columns = ca_features.columns
train_shap_df_two = train_shap_df_two.add_suffix("_shap_two")

# On the target shift
train_one_target_shift_df = pd.DataFrame(train_one_target_shift, index=[0]).T
train_one_target_shift_df.columns = ["target"]
train_two_target_shift_df = pd.DataFrame(train_two_target_shift, index=[0]).T
train_two_target_shift_df.columns = ["target"]
# On the target
tpr_one = np.array(list(tpr_one.values()))
tpr_two = np.array(list(tpr_two.values()))
# %%
## OOD State loop
for state in tqdm(states, desc="States", position=0):
    print(state)

    # Load and process data
    tx_data = data_source.get_data(states=["HI"], download=True)
    tx_features, tx_labels, tx_group = ACSIncome.df_to_numpy(tx_data)
    tx_features = pd.DataFrame(tx_features, columns=ACSIncome.features)

    # Lets add the target to ease the sampling
    tx_full = tx_features.copy()
    tx_full["group"] = tx_group
    tx_full["target"] = tx_labels

    # Loop to create training data
    for i in tqdm(range(0, int(ITERS / 10)), leave=False, desc="Bootstrap", position=1):
        row_ood_one = []
        row_ood_two = []
        row_shap_ood_one = []
        row_shap_ood_two = []

        # Sampling
        aux_ood = tx_full.sample(n=SAMPLE_FRAC, replace=True)

        # OOD performance calculation
        preds_ood = model.predict_proba(aux_ood.drop(columns=["target", "group"]))[:, 1]
        performance_ood[i] = train_error - roc_auc_score(
            aux_ood.target.values, preds_ood
        )
        ## Fairness
        white_tpr = np.mean(preds_ood[(aux_ood.target == 1) & (aux_ood.group == 1)])
        black_tpr = np.mean(preds_ood[(aux_ood.target == 1) & (aux_ood.group == 2)])
        tpr_ood_one[i] = white_tpr
        tpr_ood_two[i] = black_tpr

        # Shap values calculation OOD
        shap_values_ood = explainer(aux_ood.drop(columns=["target", "group"]))
        shap_values_ood = pd.DataFrame(
            shap_values_ood.values, columns=ca_features.columns
        )
        shap_values_ood["group"] = aux_ood.group.values

        for feat in ca_features.columns:
            # OOD
            ks_ood_one = wasserstein_distance(
                ca_features[ca_group == 1][feat], aux_ood[aux_ood["group"] == 1][feat]
            )
            ks_ood_two = wasserstein_distance(
                ca_features[ca_group == 2][feat], aux_ood[aux_ood["group"] == 2][feat]
            )

            sh_ood_one = wasserstein_distance(
                shap_test[ca_group == 1][feat],
                shap_values_ood[shap_values_ood["group"] == 1][feat],
            )
            sh_ood_two = wasserstein_distance(
                shap_test[ca_group == 2][feat],
                shap_values_ood[shap_values_ood["group"] == 2][feat],
            )

            row_ood_one.append(ks_ood_one)
            row_ood_two.append(ks_ood_two)
            row_shap_ood_one.append(sh_ood_one)
            row_shap_ood_two.append(sh_ood_two)
        # Target shift
        ks_one_target = wasserstein_distance(
            ca_labels[ca_group == 1], aux[aux["group"] == 1]["target"]
        )
        ks_two_target = wasserstein_distance(
            ca_labels[ca_group == 2], aux[aux["group"] == 2]["target"]
        )

        train_one_target_shift_ood[i] = ks_one_target
        train_two_target_shift_ood[i] = ks_two_target

        # Save OOD
        train_shap_ood_one[i] = row_shap_ood_one
        train_shap_ood_two[i] = row_shap_ood_two
        train_ood_one[i] = row_ood_one
        train_ood_two[i] = row_ood_two

    # Save results
    ## Test (previous OOD)
    train_df_ood_one = pd.DataFrame(train_ood_one).T
    train_df_ood_one.columns = ca_features.columns
    train_df_ood_one = train_df_ood_one.add_suffix("_one")
    train_df_ood_two = pd.DataFrame(train_ood_two).T
    train_df_ood_two.columns = ca_features.columns
    train_df_ood_two = train_df_ood_two.add_suffix("_two")

    train_shap_df_ood_one = pd.DataFrame(train_shap_ood_one).T
    train_shap_df_ood_one.columns = ca_features.columns
    train_shap_df_ood_one = train_shap_df_ood_one.add_suffix("_shap_one")
    train_shap_df_ood_two = pd.DataFrame(train_shap_ood_two).T
    train_shap_df_ood_two.columns = ca_features.columns
    train_shap_df_ood_two = train_shap_df_ood_two.add_suffix("_shap_two")

    # Target Shift
    train_one_target_shift_df_ood = pd.DataFrame(
        train_one_target_shift_ood, index=[0]
    ).T
    train_one_target_shift_df_ood.columns = ["target"]
    train_two_target_shift_df_ood = pd.DataFrame(
        train_two_target_shift_ood, index=[0]
    ).T
    train_two_target_shift_df_ood.columns = ["target"]
    # On the target
    try:
        tpr_ood_one = np.array(list(tpr_ood_one.values()))
        tpr_ood_two = np.array(list(tpr_ood_two.values()))
    except:
        pass

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
    estimators["SVM"] = Pipeline([("scaler", StandardScaler()), ("model", SVR(C=0.01))])
    estimators["MLP"] = Pipeline(
        [("scaler", StandardScaler()), ("model", MLPRegressor(random_state=0))]
    )

    ## Loop over different G estimators

    # Performance
    loop_estimators_fairness(
        estimator_set=estimators,
        normal_data=train_df_one,
        normal_data_ood=train_df_ood_one,
        shap_data=train_shap_df_one,
        shap_data_ood=train_shap_df_ood_one,
        target_shift=train_two_target_shift_df,
        target_shift_ood=train_two_target_shift_df_ood,
        performance_ood=tpr_tr_one - tpr_ood_one,
        target=tpr_tr_one - tpr_one,
        state=state,
        error_type="fairness_one",
    )
    # Fairness
    loop_estimators_fairness(
        estimator_set=estimators,
        normal_data=train_df_two,
        normal_data_ood=train_df_ood_two,
        shap_data=train_shap_df_two,
        shap_data_ood=train_shap_df_ood_two,
        target_shift=train_one_target_shift_df,
        target_shift_ood=train_one_target_shift_df_ood,
        performance_ood=tpr_tr_two - tpr_ood_two,
        target=tpr_tr_two - tpr_two,
        state=state,
        error_type="fairness_two",
    )

# %%
