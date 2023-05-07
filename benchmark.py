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
from xgboost import XGBRegressor, XGBClassifier

# Seeding
np.random.seed(0)
random.seed(0)
# %%
# Load data
state = "CA"
year = "2014"
N_b = 2
boots_size = 0.632
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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=boots_size, random_state=0
)
# Random assign
Z_train = X_train["group"]
X_train = X_train.drop(["group"], axis=1)
Z_test = X_test["group"]
X_test = X_test.drop(["group"], axis=1)
# %%
# Import models
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.gaussian_process import (
    GaussianProcessClassifier,
    GaussianProcessRegressor,
)


# Regressors
fmodels = [
    # LogisticRegression(),
    XGBRegressor(),
    LinearSVC(),
    SVC(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    MLPRegressor(),
    KNeighborsRegressor(),
    GaussianProcessRegressor(),
]
# Classifiers
gmodels = [
    LogisticRegression(),
    XGBClassifier(),
    SVC(kernel="linear", probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    MLPClassifier(),
    KNeighborsClassifier(),
]


aucs = []
for model in fmodels:
    for gmodel in gmodels:
        print(model.__class__.__name__, gmodel.__class__.__name__)
        try:
            audit = ExplanationAudit(
                model=model,
                gmodel=gmodel,
            )

            audit.fit_pipeline(X=X_train, y=y_train, z=Z_train)
        except:
            audit = ExplanationAudit(
                model=model,
                gmodel=gmodel,
                masker=True,
                data_masker=X_train,
            )
            audit.fit_pipeline(X=X_train, y=y_train, z=Z_train)
        # Save results
        auc = roc_auc_score(y_test, audit.predict_proba(X_test)[:, 1])

        aucs.append([auc, model.__class__.__name__, gmodel.__class__.__name__])
# %%
aucs_df = pd.DataFrame(aucs, columns=["auc", "fmodel", "gmodel"])
aucs_df
# %%
