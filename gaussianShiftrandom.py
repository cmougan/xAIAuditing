# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingRegressor
import shap

# %%
np.random.random(4)
# %%
## Create variables
### Normal
sigma = 5
mean = [0, 0]
cov = [[sigma, 0], [0, sigma]]
samples = 50_000
x1, x2 = np.random.multivariate_normal(mean, cov, samples).T
x3 = np.random.random(samples)
# Different values
mean = [0, 0]
out = 1
cov = [[sigma, out], [out, sigma]]
x11, x22 = np.random.multivariate_normal(mean, cov, samples).T
x33 = np.random.random(samples)
# %%
## Plotting
plt.figure()
sns.histplot(x1, color="r")
sns.histplot(x11)


# %%

plt.figure()
plt.scatter(x1, x2, label="Init")
plt.scatter(x11, x22, label="Different")

# %%
df = pd.DataFrame(data=[x1, x2, x3]).T
df.columns = ["Var%d" % (i + 1) for i in range(df.shape[1])]
# df["target"] = np.where(df["Var1"] * df["Var2"] > 0, 1, 0)
df["target"] = df["Var1"] * df["Var2"]
# %%
## Fit our ML model
X_tr, X_te, y_tr, y_te = train_test_split(df.drop(columns="target"), df[["target"]])
# %%
model = GradientBoostingRegressor()
model.fit(X_tr, y_tr)
y_hat = model.predict(X_te)
# %%
## Real explanation
explainer = shap.Explainer(model)
shap_values = explainer(X_te)
exp = pd.DataFrame(
    data=shap_values.values, columns=["Shap%d" % (i + 1) for i in range(3)]
)
# %%
## Fake explanation
fake = pd.DataFrame(data=[x11, x22, x33]).T
y_hat1 = model.predict(fake)
shap_values = explainer(fake)
exp_fake = pd.DataFrame(
    data=fake.values, columns=["Shap%d" % (i + 1) for i in range(3)]
)
# %%
print("Feat 1")
print(ks_2samp(exp_fake["Shap1"], exp["Shap1"]))
print(ks_2samp(x11, x1))
# %%
print("Feat 2")
print(ks_2samp(exp_fake["Shap2"], exp["Shap2"]))
print(ks_2samp(x22, x2))
# %%
print("Feat 3")
print(ks_2samp(exp_fake["Shap3"], exp["Shap3"]))
print(ks_2samp(x33, x3))
# %%
print("Target")
print(ks_2samp(y_hat, y_hat1))

# %%
