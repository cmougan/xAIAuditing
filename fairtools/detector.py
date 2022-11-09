from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import shap
import pandas as pd


class ExplanationAudit(BaseEstimator, ClassifierMixin):
    """ """

    def __init__(self, model, gmodel):
        self.model = model
        self.gmodel = gmodel
        self.explainer = None

        # Supported F Models
        self.supported_tree_models = ["XGBClassifier", "XGBRegressor"]
        self.supported_linear_models = [
            "LogisticRegression",
            "LinearRegression",
            "Ridge",
            "Lasso",
        ]
        self.supported_models = (
            self.supported_tree_models + self.supported_linear_models
        )
        # Supported detectors
        self.supported_linear_detectors = [
            "LogisticRegression",
        ]
        self.supported_tree_detectors = ["XGBClassifier"]
        self.supported_detectors = (
            self.supported_linear_detectors + self.supported_tree_detectors
        )

        # Check if models are supported
        if self.get_model_type() not in self.supported_models:
            raise ValueError(
                "Model not supported. Supported models are: {} got {}".format(
                    self.supported_models, self.model.__class__.__name__
                )
            )
        if self.get_gmodel_type() not in self.supported_detectors:
            raise ValueError(
                "gmodel not supported. Supported models are: {} got {}".format(
                    self.supported_detectors, self.gmodel.__class__.__name__
                )
            )

    def get_gmodel_type(self):
        if self.gmodel.__class__.__name__ == "Pipeline":
            return self.gmodel.steps[-1][1].__class__.__name__
        else:
            return self.gmodel.__class__.__name__

    def get_model_type(self):
        if self.model.__class__.__name__ == "Pipeline":
            return self.model.steps[-1][1].__class__.__name__
        else:
            return self.model.__class__.__name__

    def get_split_data(self, X, y, n1=0.6, n2=0.5):
        self.X_tr, X_val, self.y_tr, y_val = train_test_split(
            X, y, random_state=0, test_size=0.6
        )
        self.X_val, self.X_te, self.y_val, self.y_te = train_test_split(
            X_val, y_val, random_state=0, test_size=0.5
        )
        # Check number of classes present in label
        if len(set(self.y_tr)) <= 1:
            raise ValueError("Train set has only one class")
        if len(set(self.y_val)) <= 1:
            raise ValueError("Validation set has only one class")
        if len(set(self.y_te)) <= 1:
            raise ValueError("Test set has only one class")

        return self.X_tr, self.X_val, self.X_te, self.y_tr, self.y_val, self.y_te

    def fit(self, X, y, Z):

        # Check that X and y have correct shape
        check_X_y(X, y)
        self.Z = Z

        # Split data intro train, validation and test
        _ = self.get_split_data(X, y)

        # Extract prottected att. and remove from data
        self.Z_tr = self.X_tr[self.Z]
        self.Z_val = self.X_val[self.Z]
        self.Z_te = self.X_te[self.Z]
        self.X_tr = self.X_tr.drop(self.Z, axis=1)
        self.X_val = self.X_val.drop(self.Z, axis=1)
        self.X_te = self.X_te.drop(self.Z, axis=1)

        # Fit model F
        self.fit_model(self.X_tr, self.y_tr)

        # Get explanations
        self.S_val = self.get_explanations(self.X_val)

        # Fit model G
        self.fit_audit_detector(self.S_val, self.Z_val)

        return self

    def predict(self, X):
        if self.Z in X.columns:
            X = X.drop(self.Z, axis=1)
        return self.gmodel.predict(self.get_explanations(X))

    def predict_proba(self, X):
        if self.Z in X.columns:
            X = X.drop(self.Z, axis=1)
        return self.gmodel.predict_proba(self.get_explanations(X))

    def explanation_predict(self, X):
        return self.gmodel.predict(X)

    def explanation_predict_proba(self, X):
        return self.gmodel.predict_proba(X)

    def fit_model(self, X, y):
        self.model.fit(X, y)

    def fit_audit_detector(self, X, y):
        self.gmodel.fit(X, y)

    def get_explanations(self, X):
        # Determine the type of SHAP explainer to use
        if self.get_model_type() in self.supported_tree_models:
            self.explainer = shap.Explainer(self.model)
        elif self.get_model_type() in self.supported_linear_models:
            self.explainer = shap.LinearExplainer(
                self.model, X, feature_dependence="correlation_dependent"
            )
        else:
            raise ValueError(
                "Model not supported. Supported models are: {}, got {}".format(
                    self.supported_models, self.model.__class__.__name__
                )
            )

        shap_values = self.explainer(X)
        # Name columns
        if isinstance(X, pd.DataFrame):
            columns_name = X.columns
        else:
            columns_name = ["Shap%d" % (i + 1) for i in range(X.shape[1])]

        exp = pd.DataFrame(
            data=shap_values.values,
            columns=columns_name,
        )
        return exp

    def get_auc_f_val(self):
        """
        TODO Case of F being a classifier
        """
        return roc_auc_score(self.y_val, self.model.predict(self.X_val))

    def get_auc_val(self):
        """
        Returns the AUC of the validation set

        """
        return roc_auc_score(self.y_te, self.predict_proba(self.X_te)[:, 1])

    def get_coefs(self):
        if self.gmodel.__class__.__name__ == "Pipeline":
            if (
                self.gmodel.steps[-1][1].__class__.__name__
                in self.supported_linear_models
            ):
                return self.gmodel.steps[-1][1].coef_
            else:
                raise ValueError(
                    "Pipeline model not supported. Supported models are: {}, got {}".format(
                        self.supported_linear_models,
                        self.gmodel.steps[-1][1].__class__.__name__,
                    )
                )
        else:
            return self.get_linear_coefs()

    def get_linear_coefs(self):
        if self.gmodel.__class__.__name__ in self.supported_linear_models:
            return self.gmodel.coef_
        else:
            raise ValueError(
                "Detector model not supported. Supported models ar linear: {}, got {}".format(
                    self.supported_linear_detector, self.model.__class__.__name__
                )
            )
