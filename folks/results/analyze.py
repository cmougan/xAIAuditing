# %%
import pandas as pd

# %%
df = pd.read_csv("all_results.csv")

df = df[df["error_type"] == "performance"]
df = df.drop(columns="error_ood")
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
df
# %%
df = df[df["error_type"] == "performance"]
# %%
