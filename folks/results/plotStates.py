# %%
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

df = pd.read_csv("all_results.csv")
# %%
aux = df[df["error_type"] == "performance"]
aux = aux[aux["estimator"] == "Linear"]
aux = aux[aux["data"] == "Only Shap"]
fig = px.choropleth(
    aux.groupby(["state"]).mean().reset_index(),
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
    aux.groupby(["state"]).mean().reset_index(),
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

# %%
