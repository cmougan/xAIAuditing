# %%
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from collections import defaultdict

from pyparsing import col

df = pd.read_csv("all_results.csv")
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

aux = df[df["error_type"] == "fairness"]
aux = aux[aux["estimator"] == "Linear"]
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
    data = aux_state[aux_state["data"] == "Only Data"].error_ood.values
    shap = aux_state[aux_state["data"] == "Only Shap"].error_ood.values
    both = aux_state[aux_state["data"] == "Data + Shap"].error_ood.values
    # Dummy
    aux_state = aux[(aux["state"] == state) & (aux["estimator"] == "Dummy")]
    dummy = aux_state.error_ood.mean()
    d = {"data": data, "shap": shap, "both": both, "dummy": dummy}

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
fig.write_image("../../images/best_method.svg", format="svg")

# %%
