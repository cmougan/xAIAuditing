# %%
from pmlb import classification_dataset_names, regression_dataset_names
from fairtools.benchmark import benchmark_experiment
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import warnings
import pandas as pd

warnings.filterwarnings("ignore")


# %%
regression_dataset_names_sample = regression_dataset_names[:5]
# %%
regression_dataset_names[-1:]
# %%

modelitos = [
    GradientBoostingRegressor(),
]
for m in modelitos:
    benchmark_experiment(
        datasets=["titanic"],
        model=m,
        classification="explainableAI",
    )
# %%