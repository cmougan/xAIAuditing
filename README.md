[![black](https://img.shields.io/badge/code%20style-black-000000.svg?style=plastic)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?color=g&style=plastic)](https://opensource.org/licenses/MIT)

# Discrimination Audits via the Explanation Space


This repository contains the code for the paper Discrimination Audits via the Explanation Space, which is available on ...


```python
from nobias import ExplanationAudit
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor
```


# Tutorial with Synthetic Dataset

```python
# Create synthetic data
N = 5_000
x1 = np.random.normal(1, 1, size=N)
x2 = np.random.normal(1, 1, size=N)
x34 = np.random.multivariate_normal([1, 1], [[1, 0.5], [0.5, 1]], size=N)
x3 = x34[:, 0]
x4 = x34[:, 1]
# Binarize protected attribute - Named var4
x4 = np.where(x4 > np.mean(x4), 1, 0)
X = pd.DataFrame([x1, x2, x3, x4]).T
X.columns = ["var%d" % (i + 1) for i in range(X.shape[1])]
y = 1 / (1 + np.exp(-(x1 + x2 + x3) / 3))
```

```python
detector = ExplanationAudit(model=XGBRegressor(), gmodel=LogisticRegression())
detector.fit(X, y, Z="var4")
detector.get_auc_val()
```

```python
detector.get_auc_val()
#0.7
```
```python
coefs = detector.get_coefs()
coefs = pd.DataFrame(coefs, index=X.columns[:-1], columns=["coef"]).sort_values("coef", ascending=False)
coefs.plot(kind="bar")
```

<p align="center">
  <img width="616" src="https://raw.githubusercontent.com/cmougan/xAIAuditing/master/images/coefs_synth.png" />
</p>

## Tutorial on Real Dataset
```python
# Load data from the folktables package
from folktables import ACSDataSource, ACSIncome

data_source = ACSDataSource(survey_year="2014", horizon="1-Year", survey="person")
acs_data = data_source.get_data(states=["CA"], download=True)
ca_features, ca_labels, ca_group = ACSIncome.df_to_numpy(acs_data)
X, y, _ = ACSIncome.df_to_numpy(acs_data)
X = pd.DataFrame(X, columns=ACSIncome.features)
# White vs ALL
X["RAC1P"] = np.where(X["RAC1P"] == 1, 1, 0)
```

```python
detector = ExplanationAudit(
        model=XGBRegressor(random_state=0), gmodel=LogisticRegression()
    )
detector.fit(X, y, Z="RAC1P")
```

```python
detector.get_auc_val()
#0.7
```
```python
coefs = detector.get_coefs()
coefs = pd.DataFrame(coefs, index=X.columns[:-1], columns=["coef"]).sort_values("coef", ascending=False)
coefs.plot(kind="bar")
```

<p align="center">
  <img width="616" src="https://raw.githubusercontent.com/cmougan/xAIAuditing/master/images/coefs_real.png" />
</p>
