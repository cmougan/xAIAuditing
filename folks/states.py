# %%
from cProfile import label
from folktables import ACSDataSource, ACSIncome
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
import pandas as pd
from collections import defaultdict
from xgboost import XGBRegressor, XGBClassifier
from scipy.stats import kstest
import shap
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")

import random

random.seed(0)
# %%
# Load data
data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
ca_data = data_source.get_data(states=["CA"], download=True)
ca_features, ca_labels, ca_group = ACSIncome.df_to_numpy(ca_data)
## Conver to DF
ca_features = pd.DataFrame(ca_features, columns=ACSIncome.features)

# %%
splitter = int(ca_features.shape[0] / 2)
ca_features_train = ca_features[:splitter]
ca_features_test = ca_features[splitter:]
ca_group_train = ca_group[:splitter]
ca_group_test = ca_group[splitter:]
ca_labels_train = ca_labels[:splitter]
ca_labels_test = ca_labels[splitter:]
# %%
# Modeling
model = XGBClassifier()

# Train on CA data
preds_ca_train = cross_val_predict(
    model, ca_features_train, ca_labels_train, method="predict_proba", cv=3
)[:, 1]
model.fit(ca_features, ca_labels)

# Test on MI data
preds_ca_test = model.predict_proba(ca_features_test)[:, 1]

# %%
##Fairness
white_tpr = np.mean(preds_ca_train[(ca_labels_train == 1) & (ca_group_train == 1)])
black_tpr = np.mean(preds_ca_train[(ca_labels_train == 1) & (ca_group_train == 2)])
print("EOF", white_tpr - black_tpr)

## Model performance
print("AUC", roc_auc_score(ca_labels_train, preds_ca_train))
# %%
## Demographic parity
plt.figure()
ks = kstest(
    preds_ca_train[(ca_group_train == 1)], preds_ca_train[(ca_group_train == 2)]
)
plt.title(
    "Distribution of predictions; KS test: {:.2f} with pvalue {:.2e}".format(
        ks.statistic, ks.pvalue
    )
)
sns.kdeplot(preds_ca_train[(ca_group_train == 1)], shade=True, label="White")
sns.kdeplot(preds_ca_train[(ca_group_train == 2)], shade=True, label="Black")
plt.legend()
plt.show()

# %%
explainer = shap.Explainer(model)
shap_values = explainer(ca_features_test)
ca_shap = pd.DataFrame(shap_values.values, columns=ca_features.columns)

# %%

X_tr, X_te, y_tr, y_te = train_test_split(
    ca_shap[(ca_group_test == 1) | (ca_group_test == 2)],
    ca_group_test[(ca_group_test == 1) | (ca_group_test == 2)],
    random_state=0,
    test_size=0.2,
    stratify=ca_group_test[(ca_group_test == 1) | (ca_group_test == 2)],
)
# %%
m = LogisticRegression()
m.fit(X_tr, y_tr)
print("AUC shap", roc_auc_score(y_te, m.predict_proba(X_te)[:, 1], multi_class="ovr"))

# %%
explainer = shap.LinearExplainer(m, X_tr, feature_dependence="correlation_dependent")
shap_test = explainer(X_te)
# shap_test = pd.DataFrame(shap_test.values, columns=ca_features.columns)
# %%
shap.plots.waterfall(shap_values[0])
# %%
shap.plots.bar(shap_values)
# %%
shap.plots.beeswarm(shap_values)
# %%
