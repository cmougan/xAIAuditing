# %%
# Load data from the folktables package
from folktables import ACSDataSource, ACSIncome
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")
data_source = ACSDataSource(survey_year="2014", horizon="1-Year", survey="person")
acs_data = data_source.get_data(states=["CA"], download=True)
ca_features, ca_labels, ca_group = ACSIncome.df_to_numpy(acs_data)
X, y, _ = ACSIncome.df_to_numpy(acs_data)
X = pd.DataFrame(X, columns=ACSIncome.features)
# White vs ALL
X["RAC1P"] = np.where(X["RAC1P"] == 1, 1, 0)
# %%
from fairtools.detector import ExplanationAudit
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression

detector = ExplanationAudit(
    model=XGBRegressor(random_state=0), gmodel=LogisticRegression()
)
detector.fit(X, y, Z="RAC1P")
# %%
detector.get_auc_val()
# %%
coefs = detector.gmodel.coef_[0]
coefs = pd.DataFrame(coefs, index=X.columns[:-1], columns=["coef"]).sort_values(
    "coef", ascending=False
)

coefs.plot(kind="bar")
# %%
plt.figure(figsize=(10, 10))
coefs.plot(kind="bar")
plt.tight_layout()
plt.savefig("images/coefs_real.png")
plt.show()

# %%
import seaborn as sns

plt.figure(figsize=(10, 10))
sns.heatmap(coefs, annot=True)
plt.show()
# %%
