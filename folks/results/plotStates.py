# %%
import pandas as pd
import plotly.express as px

df = pd.read_csv(r"state_market_tracker.tsv000.gz", sep="\t")
df = df[["period_begin", "state", "state_code", "property_type", "median_sale_price"]]
df = df[(df["property_type"] == "Single Family Residential")]
df.rename({"median_sale_price": "Median Sales Price ($)"}, axis=1, inplace=True)
df["period_begin"] = pd.to_datetime(df["period_begin"]).dt.date.astype(str)
df = df.sort_values(
    "period_begin"
)  # Make sure you sort the time horizon column in ascending order because this column is in random order in the raw dataset

fig = px.choropleth(
    df,
    locations="state_code",
    locationmode="USA-states",
    color="Median Sales Price ($)",
    color_continuous_scale="Viridis_r",
    scope="usa",
)  # make sure 'period_begin' is string type and sorted in ascending order

fig.show()
# %%
df
# %%
data = pd.read_csv("all_results_13states.csv")
# %%
aux = data.groupby(["state"]).mean().reset_index()
# %%
df
# %%

fig = px.choropleth(
    aux,
    locations="state",
    locationmode="USA-states",
    color="error_te",
    color_continuous_scale="Viridis_r",
    scope="usa",
)  # make sure 'period_begin' is string type and sorted in ascending order

fig.show()
# %%
aux2 = df.groupby("state_code").mean().reset_index()
# %%
fig = px.choropleth(
    aux2,
    locations="state_code",
    locationmode="USA-states",
    color="Median Sales Price ($)",
    color_continuous_scale="Viridis_r",
    scope="usa",
)  # make sure 'period_begin' is string type and sorted in ascending order

fig.show()
# %%
us_states = df.state_code.unique()
# %%

i = 0
for state in data.state.unique():
    if state not in us_states:
        i = i + 1
        print(state)
# %%
"SD" in us_states
# %%
