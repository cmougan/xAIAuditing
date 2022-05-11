# %%
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from pyparsing import col

df = pd.read_csv("all_results_performnnce.csv")
# %%
df.data = np.where(df.data == "Only Data", "Distribution Shift", df.data)
df.data = np.where(df.data == "Only Shap", "Explanation Shift", df.data)
df.data = np.where(df.data == "Data + Shap", "Exp+Dist Shift", df.data)
# %%
aux = df[df["error_type"] == "performance"]
aux = aux[aux["estimator"] == "Linear"]
aux = aux[aux["data"] == "Only Shap"]
fig = px.choropleth(
    aux.groupby(["state"]).min().reset_index(),
    locations="state",
    locationmode="USA-states",
    color="error_ood",
    color_continuous_scale="Reds",
    scope="usa",
    hover_name="state",
    hover_data=["error_ood"],
)
fig.show()
fig.write_image("../../images/performanceUS.svg", format="svg")
fig.write_image("../../images/performanceUS.png")
# %%
aux = df[df["error_type"] == "fairness"]
aux = aux[(aux["estimator"] == "Linear") & (aux["estimator"] == "XGBoost")]
aux = aux[aux["data"] == "Only Data"]
fig = px.choropleth(
    aux.groupby(["state"]).min().reset_index(),
    locations="state",
    locationmode="USA-states",
    color="error_ood",
    color_continuous_scale="Reds",
    scope="usa",
    hover_name="state",
    hover_data=["error_ood"],
)
fig.show()


# %%
aux = df[df["error_type"] == "performance"]
aux = aux[(aux["estimator"] == "Linear") | (aux["estimator"] == "Dummy")]

best = []
for state in aux["state"].unique():
    aux_state = aux[(aux["state"] == state) & (aux["estimator"] == "Linear")]
    # Estimators
    data = aux_state[aux_state["data"] == "Distribution Shift"].error_ood.values
    shap = aux_state[aux_state["data"] == "Explanation Shift"].error_ood.values
    both = aux_state[aux_state["data"] == "Exp+Dist Shift"].error_ood.values
    # Dummy
    aux_state = aux[(aux["state"] == state) & (aux["estimator"] == "Dummy")]
    dummy = aux_state.error_ood.mean()
    d = {
        "Distribution Shift": data,
        "Explanation Shift": shap,
        "Exp+Dist Shift": both,
        "dummy": dummy,
    }

    best.append([state, min(d, key=d.get)])

best = pd.DataFrame(best, columns=["state", "data"])
# %%
fig = px.choropleth(
    best,
    locations="state",
    locationmode="USA-states",
    color="data",
    # color_continuous_scale="Reds",
    scope="usa",
    hover_name="state",
    # hover_data=["error_ood"],
)
fig.show()
fig.write_image("../../images/best_method.png")

# %%
