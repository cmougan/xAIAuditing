import shap
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score


def shap_detector(data1, data2, verbosity=False):

    shapX1 = pd.DataFrame(data1)
    shapX1.columns = ["var%d" % (i + 1) for i in range(shapX1.shape[1])]
    shapX1["target"] = 0
    shapX2 = pd.DataFrame(data2)
    shapX2.columns = ["var%d" % (i + 1) for i in range(shapX2.shape[1])]
    shapX2["target"] = 1
    shapALL = shapX1.append(shapX2)
    X_train, X_test, y_train, y_test = train_test_split(
        shapALL.drop(columns=["target"]),
        shapALL.target,
        test_size=0.33,
        random_state=42,
    )
    det = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, verbosity=0)
    det.fit(X_train, y_train)

    if verbosity:
        print("AUC Train:", roc_auc_score(y_train, det.predict_proba(X_train)[:, 1]))
        print("AUC Test:", roc_auc_score(y_test, det.predict_proba(X_test)[:, 1]))
    return roc_auc_score(y_test, det.predict_proba(X_test)[:, 1])
