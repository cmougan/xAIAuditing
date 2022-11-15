[![black](https://img.shields.io/badge/code%20style-black-000000.svg?style=plastic)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?color=g&style=plastic)](https://opensource.org/licenses/MIT)

# Discrimination Audits via the Explanation Space


```python
# Load data from the folktables package
from folktables import ACSDataSource, ACSIncome

data_source = ACSDataSource(survey_year="2014", horizon="1-Year", survey="person")
acs_data = data_source.get_data(states=["CA"], download=True)
ca_features, ca_labels, ca_group = ACSIncome.df_to_numpy(acs_data)
# Create a Dataframe
ca_features = pd.DataFrame(ca_features, columns=ACSIncome.features)
ca_features["group"] = ca_group
ca_features["label"] = ca_labels
# Binarize the protected attribute
ca_features["group"] = np.where(ca_features["group"] == 1, 1, 0)

# Split data
X = ca_features.drop(["label", "RAC1P"], axis=1)
y = ca_features["label"]
```

```python
detector = ExplanationAudit(
        model=XGBRegressor(random_state=0), gmodel=LogisticRegression()
    )

detector.fit(X, y, Z="group")
```

```python
detector.get_auc_val()
#0.7
```
```python
detector.get_coefs()
```
