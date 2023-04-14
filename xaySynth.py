# %%
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier, XGBRegressor
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from nobias import ExplanationAudit

plt.style.use("seaborn-whitegrid")
# %%
# TODO -- Add two variables that contribute.
N = 5_000
CASE_A = False
dp = []
res = []
exp_evolution = pd.DataFrame()
coefs = []
params = np.linspace(0, 1, 20)
for gamma in params:
    x1 = np.random.normal(1, 1, size=N)
    x2 = np.random.normal(1, 1, size=N)

    x36 = np.random.multivariate_normal([1, 1], [[1, gamma], [gamma, 1]], size=N)
    x3 = x36[:, 0]
    x6 = x36[:, 1]

    x45 = np.random.multivariate_normal(
        [1, 1], [[1, gamma / 2], [gamma / 2, 1]], size=N
    )
    x4 = x45[:, 0]
    x5 = x45[:, 1]

    x5 = x5 + x6

    # Binarize protected attribute
    x5 = np.where(x5 > np.mean(x5), 1, 0)

    X = pd.DataFrame([x1, x2, x3, x4, x5]).T
    X.columns = ["var%d" % (i + 1) for i in range(X.shape[1])]

    # Different exp -- Remember to change the name of the visualization
    # Case A
    if CASE_A:
        y = (x1 + x2 + x3 + x4) / 3
    else:
        # Case B
        y = (x1 + x2) / 3

    y = 1 / (1 + np.exp(-y))

    # Train test hold out split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=0.33,
        random_state=42,
    )
    X_te, X_val, y_te, y_val = train_test_split(
        X_te,
        y_te,
        test_size=0.5,
        random_state=42,
    )

    Z_tr = X_tr.var5.values
    Z_te = X_te.var5.values
    Z_val = X_val.var5.values
    X_tr = X_tr.drop(["var5"], axis=1)
    X_te = X_te.drop(["var5"], axis=1)
    X_val = X_val.drop(["var5"], axis=1)

    # Train model
    model = XGBRegressor()
    model.fit(X_tr, y_tr)
    # Inspector
    inspector = ExplanationAudit(
        model, gmodel=LogisticRegression(penalty="l1", solver="saga")
    )
    inspector.fit_inspector(X_te, Z_te)

    print(roc_auc_score(Z_val, model.predict(X_val)))

    coefs.append(inspector.inspector.coef_)


# %%
# convert to dataframe
coefs_df = pd.DataFrame(coefs[0])

for i in range(len(coefs)):
    if i > 0:
        coefs_df = pd.concat([coefs_df, pd.DataFrame(coefs[i])], axis=0)
coefs_df.columns = ["var1", "var2", "var3", "var4"]
# %%
# Plot three coefficients
plt.plot(np.linspace(0, 1, 20), coefs_df["var1"], label="var1")
plt.fill_between(
    np.linspace(0, 1, 20),
    coefs_df["var1"] - 0.5 * coefs_df["var1"].std(),
    coefs_df["var1"] + 0.5 * coefs_df["var1"].std(),
    alpha=0.2,
)

plt.plot(np.linspace(0, 1, 20), coefs_df["var2"], label="var2")
plt.fill_between(
    np.linspace(0, 1, 20),
    coefs_df["var2"] - 0.5 * coefs_df["var2"].std(),
    coefs_df["var2"] + 0.5 * coefs_df["var2"].std(),
    alpha=0.2,
)


plt.plot(np.linspace(0, 1, 20), coefs_df["var3"], label="var3")
plt.fill_between(
    np.linspace(0, 1, 20),
    coefs_df["var3"] - 0.5 * coefs_df["var3"].std(),
    coefs_df["var3"] + 0.5 * coefs_df["var3"].std(),
    alpha=0.2,
)

plt.plot(np.linspace(0, 1, 20), coefs_df["var4"], label="var4")
plt.fill_between(
    np.linspace(0, 1, 20),
    coefs_df["var4"] - 0.5 * coefs_df["var4"].std(),
    coefs_df["var4"] + 0.5 * coefs_df["var4"].std(),
    alpha=0.2,
)

plt.xlabel("Correlation")
plt.legend()

plt.ylabel("Coefficients of the inspector")
plt.title(
    "Coefficient evolution with correlation for case {0}".format(
        "Indirect" if CASE_A else "Uninformative"
    )
)
plt.ylim(-0.5, 35)
plt.savefig(
    "images/coef_evolution{0}.pdf".format("A" if CASE_A else "B"), bbox_inches="tight"
)
plt.show()

# %%
