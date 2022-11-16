[![black](https://img.shields.io/badge/code%20style-black-000000.svg?style=plastic)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?color=g&style=plastic)](https://opensource.org/licenses/MIT)

# Discrimination Audits via the Explanation Space


```python
# Load data from the folktables package
from folktables import ACSDataSource, ACSIncome
import pandas as pd
import numpy as np
data_source = ACSDataSource(survey_year="2014", horizon="1-Year", survey="person")
acs_data = data_source.get_data(states=["CA"], download=True)
ca_features, ca_labels, ca_group = ACSIncome.df_to_numpy(acs_data)
X, y, _ = ACSIncome.df_to_numpy(acs_data)
X = pd.DataFrame(X, columns=ACSIncome.features)
# White vs ALL
X["RAC1P"] = np.where(X["RAC1P"] == 1, 1, 0)
```

```python
from fairtools.detector import ExplanationAudit
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression
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
  <img width="616" src="https://raw.githubusercontent.com/cmougan/xAIAuditing/master/images/coefs.png" />
</p>
