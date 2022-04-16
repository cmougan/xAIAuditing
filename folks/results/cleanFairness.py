# %%
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")
# %%
files = os.listdir()

data = pd.DataFrame()
for file in files:
    if file.endswith(".csv"):
        if file != "all_results.csv":
            data = data.append(pd.read_csv(file))
# data.to_csv("all_results.csv", index=False)

# %%
data
# %%
file
# %%
data[(data["state"] == "MI") & (data["error_type"] == "fairness_one")]
# %%

# %%
