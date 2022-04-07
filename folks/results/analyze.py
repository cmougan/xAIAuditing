# %%
import pandas as pd
import numpy as np

# %%
df = pd.read_csv("all_results.csv")

df = df[df["error_type"] == "performance"]
# df = df.drop(columns="error_te")
# df = df[df['state']=='PR']
df = df[
    (df["estimator"] == "Linear")
    | (df["estimator"] == "Dummy")
    | (df["estimator"] == "XGBoost")
]

pd.pivot_table(
    df,
    columns=[
        "estimator",
        "data",
    ],
    index=["state"],
).T.sort_values(by=["estimator", "data"], ascending=False).style.highlight_min()
# %%
pd.pivot_table(
    df,
    index=[
        "estimator",
        "data",
    ],
    aggfunc=[np.mean, np.std],
).style.highlight_min()  # .T.sort_values(by=["estimator", "data"], ascending=False).style.highlight_min()
# %%
