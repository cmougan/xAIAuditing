# %%
import pandas as pd
import os

files = os.listdir("results")
files
# %%
lr = pd.read_csv("results/LinearRegression_reg.csv", index_col=0)
lasso = pd.read_csv("results/Lasso_reg.csv", index_col=0)
dt = pd.read_csv("results/DecisionTreeRegressor_reg.csv", index_col=0)
rf = pd.read_csv("results/RandomForestRegressor_reg.csv", index_col=0)
gb = pd.read_csv("results/GradientBoostingRegressor_reg.csv", index_col=0)
# %%
lr.mean()
# %%
lasso.mean()
# %%
dt.mean()
# %%
rf.mean()
# %%
gb.mean()

# %%
gb.mean()
# %%
files
# %%
