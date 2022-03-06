# %%
from pmlb import classification_dataset_names, regression_dataset_names
from benchmark import benchmark_experiment
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import warnings

warnings.filterwarnings("ignore")


# %%
regression_dataset_names_sample = regression_dataset_names[:10]
# %%

modelitos = [
    LinearRegression(),
    Lasso(),
    RandomForestRegressor(),
    DecisionTreeRegressor(),
    GradientBoostingRegressor(),
]
for m in modelitos:
    benchmark_experiment(datasets=regression_dataset_names_sample, model=m)
