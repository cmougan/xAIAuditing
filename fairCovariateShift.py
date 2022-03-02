# %%
import pandas as pd
import numpy as np
import random

random.seed(0)
from xgboost import XGBRegressor, XGBClassifier
from category_encoders.target_encoder import TargetEncoder
from category_encoders.one_hot import OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
import shap
from scipy.stats import ks_2samp
from sklearn.pipeline import Pipeline

from fairtools.utils import auc_group, metric_calculator, calculate_cm
from collections import defaultdict

# %%
df = pd.read_csv("data/adult.csv")
df = df.drop(columns=["fnlwgt", "capitalgain", "capitalloss"])
df["class"] = np.where(df["class"] == "<=50K", 1, 0)
df["sex"] = np.where(df["sex"] == "Male", 1, 0)
# %%
a = df.groupby(["race", "education"])["class"].count().reset_index()
a[a.race == "Black"].sort_values("class")
# %%
black = df[df["race"] == "Black"]
black.education.unique()
# %%
## Train Test Split
df_tr = df.loc[:24421]
df_te = df.loc[24421:]
print(df_tr.shape)
print(df_te.shape)
# %%
education_test = [
    # "Bachelors",
    # "Some-college",
    "Doctorate",
    # "Masters",
    "Prof-school",
]
# education_test = ["5th-6th"]
# %%
print(df_tr[(df_tr.race == "Black") & (df_tr.education.isin(education_test))].shape)
df_te = df_te.append(
    df_tr[(df_tr.race == "Black") & (df_tr.education.isin(education_test))]
)
df_tr = df_tr[
    ~df_tr.index.isin(
        df_tr[(df_tr.race == "Black") & (df_tr.education.isin(education_test))].index
    )
]
print(df_tr.shape)
print(df_te.shape)

# %%
X_tr = df_tr.drop(columns="class")
X_te = df_te.drop(columns="class")
y_tr = df_tr[["class"]]
y_te = df_te[["class"]]
# %%
## KS test for input data
te = TargetEncoder()
X_tr_aux = te.fit_transform(X_tr, y_tr)
X_te_aux = te.transform(X_te)
for col in X_tr.columns:
    pvalues = ks_2samp(X_tr_aux[col].values, X_te_aux[col].values).pvalue
    if pvalues < 0.1:
        print("Pvalue for {} is {}".format(col, pvalues))
# %%
# Black KS test
black_tr = te.transform(X_tr[X_tr["race"] == "Black"])
black_te = te.transform(X_te[X_te["race"] == "Black"])
for col in X_tr.columns:
    pvalues = ks_2samp(black_tr[col].values, black_te[col].values).pvalue
    if pvalues < 0.1:
        print("Pvalue for {} is {}".format(col, pvalues))

# %%
pipe = Pipeline(
    [
        ("scaler", TargetEncoder()),
        (
            "model",
            XGBClassifier(
                verbosity=0, silent=True, random_state=42, use_label_encoder=False
            ),
        ),
    ]
)
pred_train = cross_val_predict(pipe, X_tr, y_tr.values.ravel(), cv=5)
pipe.fit(X_tr, y_tr.values.ravel())
pred_test = pipe.predict(X_te)
explainer = shap.Explainer(pipe.named_steps["model"])
# %%
# Train
shap_values_train = explainer(pipe[:-1].transform(X_tr))
shap_values_train = pd.DataFrame(shap_values_train.values, columns=X_tr.columns)
# Test
shap_values_test = explainer(pipe[:-1].transform(X_te))
shap.plots.bar(shap_values_test)
shap_values_test = pd.DataFrame(shap_values_test.values, columns=X_tr.columns)

# %%
# General KS test
for col in X_tr.columns:
    pvalues = ks_2samp(
        shap_values_train[col].values, shap_values_test[col].values
    ).pvalue
    if pvalues < 0.1:
        print("Pvalue for {} is {}".format(col, pvalues))
# %%
# Black KS test
# Train
shap_values_train_black = explainer(pipe[:-1].transform(black_tr))
shap_values_train_black = pd.DataFrame(
    shap_values_train_black.values, columns=X_tr.columns
)
# Test
shap_values_test_black = explainer(pipe[:-1].transform(black_te))
shap_values_test_black = pd.DataFrame(
    shap_values_test_black.values, columns=X_tr.columns
)
for col in X_tr.columns:
    pvalues = ks_2samp(
        shap_values_train_black[col].values, shap_values_test_black[col].values
    ).pvalue
    if pvalues < 0.1:
        print("Pvalue for {} is {}".format(col, pvalues))

# %%
print("Train Error")
print(roc_auc_score(y_tr, pred_train))
print("Test Error")
print(roc_auc_score(y_te, pred_test))
# %%
res = defaultdict()
for cat, num in X_te["race"].value_counts().items():
    COL = "race"
    GROUP1 = "White"
    GROUP2 = cat
    res[cat] = [
        metric_calculator(
            modelo=pipe, data=X_te, truth=y_te, col=COL, group1=GROUP1, group2=GROUP2
        ),
        num,
    ]
# %%
# Extract metrics for each group
res_train = defaultdict()
X_tr_aux = X_tr.copy()
X_tr_aux["preds"] = pred_train
X_tr_aux["label"] = y_tr
for cat, num in X_tr_aux["race"].value_counts().items():
    GROUP1 = "White"
    # Main Group
    aux = X_tr_aux[X_tr_aux["race"] == GROUP1]
    res1 = calculate_cm(aux.preds, aux.label)
    # Minority Group
    aux = X_tr_aux[X_tr_aux["race"] == cat]
    res2 = calculate_cm(aux.preds, aux.label)
    res_train[cat] = [res1 - res2]
# %%
res
# %%
res_train

# %%
