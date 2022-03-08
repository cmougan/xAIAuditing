# %%
from folktables import ACSDataSource, ACSIncome
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from scipy.stats import kstest
import shap

# %%
data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
ca_data = data_source.get_data(states=["CA"], download=True)
mi_data = data_source.get_data(states=["MI"], download=True)
ca_features, ca_labels, _ = ACSIncome.df_to_numpy(ca_data)
mi_features, mi_labels, _ = ACSIncome.df_to_numpy(mi_data)
## Conver to DF
ca_features = pd.DataFrame(ca_features, columns=ACSIncome.features)
mi_features = pd.DataFrame(mi_features, columns=ACSIncome.features)

# %%
# Plug-in your method for tabular datasets
model = XGBClassifier()

# Train on CA data
model.fit(ca_features, ca_labels)
preds_ca = cross_val_predict(model, ca_features, ca_labels, cv=3)

# Test on MI data
preds_mi = model.predict(mi_features)
# %%
print(roc_auc_score(preds_ca, ca_labels))
print(roc_auc_score(preds_mi, mi_labels))


# %%
for feat in ca_features.columns:
    pval = kstest(ca_features[feat], mi_features[feat]).pvalue
    if pval < 0.1:
        print(feat, " is distinct ", pval)
    else:
        print(feat, " is equivalent ", pval)
# %%
# %%
# Explainability
explainer = shap.Explainer(model)
shap_values = explainer(ca_features)
ca_shap = pd.DataFrame(shap_values.values,columns=ca_features.columns)
shap_values = explainer(mi_features)
mi_shap = pd.DataFrame(shap_values.values,columns=ca_features.columns)
# %%
for feat in ca_features.columns:
    pval = kstest(ca_shap[feat], mi_shap[feat]).pvalue
    if pval < 0.1:
        print(feat, " is distinct ", pval)
    else:
        print(feat, " is equivalent ", pval)
# %%
