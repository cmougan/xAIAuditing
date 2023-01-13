from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import shap
import pandas as pd


class ExplanationAudit(BaseEstimator, ClassifierMixin):
    """
    Given a model, a dataset, and the protected attribute, we want to know if the model violates demographic parity or not and what are the features pushing for it.
    We do this by computing the shap values of the model, and then train a classifier to distinguish the protected attribute.

    Example
    -------
    >>> import pandas as pd
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.datasets import make_blobs
    >>> from tools.xaiUtils import ExplanationShiftDetector
    >>> from xgboost import XGBRegressor
    >>> from sklearn.linear_model import LogisticRegression

    >>> X, y = make_blobs(n_samples=2000, centers=2, n_features=5, random_state=0)
    >>> X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
    >>> X_ood,y_ood = make_blobs(n_samples=1000, centers=1, n_features=5, random_state=0)

    >>> detector = ExplanationShiftDetector(model=XGBRegressor(),gmodel=LogisticRegression())
    >>> detector.fit(X_tr, y_tr, X_ood)
    >>> detector.get_auc_val()
    # 0.76
    >>> detector.fit(X_tr, y_tr, X_te)
    >>> detector.get_auc_val()
    # 0.5
    """

    def __init__(
        self,
        model,
        gmodel,
        masker=False,
        space="explanation",
        algorithm: str = "auto",
        feature_perturbation=None,
    ):
        self.model = model
        self.gmodel = gmodel
        self.explainer = None
        self.space = space
        self.masker = masker
        self.algorithm = algorithm
        self.feature_perturbation = None

        # Check if space is supported
        if self.space not in ["explanation", "input", "prediction"]:
            raise ValueError(
                "space not supported. Supported spaces are: {} got {}".format(
                    ["explanation", "input", "prediction"], self.space
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

    def get_split_data(self, X, y, Z, n1=0.6, n2=0.5):
        self.X_tr, X_val, self.y_tr, y_val = train_test_split(
            X, y, random_state=0, test_size=0.6, stratify=X[Z]
        )
        self.X_val, self.X_te, self.y_val, self.y_te = train_test_split(
            X_val, y_val, random_state=0, test_size=0.5, stratify=X_val[Z]
        )
        # Check number of classes present in label
        if len(set(self.y_tr)) <= 1:
            raise ValueError("Train set has only one class")
        if len(set(self.y_val)) <= 1:
            raise ValueError("Validation set has only one class")
        if len(set(self.y_te)) <= 1:
            raise ValueError("Test set has only one class")

        # Check number or protected groups
        if len(set(self.X_tr[Z])) <= 1:
            raise ValueError("Train set has only one protected group")
        if len(set(self.X_val[Z])) <= 1:
            raise ValueError("Validation set has only one protected group")
        if len(set(self.X_te[Z])) <= 1:
            raise ValueError("Test set has only one protected group")

        return self.X_tr, self.X_val, self.X_te, self.y_tr, self.y_val, self.y_te

    def fit(self, X, y, Z):

        # Check that X and y have correct shape
        check_X_y(X, y)
        self.Z = Z

        # Split data intro train, validation and test
        _ = self.get_split_data(X, y, Z)

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

    def get_explanations(self, X, data_masker=None):
        if data_masker == None:
            data_masker = self.X_tr
        else:
            data_masker = data_masker

        if self.space == "explanation":

            if self.masker:
                self.explainer = shap.Explainer(
                    self.model, algorithm=self.algorithm, masker=data_masker
                )
            else:
                self.explainer = shap.Explainer(self.model, algorithm=self.algorithm)
            # Rewrite for the linear case
            if self.algorithm == "linear":
                self.explainer = shap.explainers.Linear(
                    self.model,
                    masker=data_masker,
                    feature_perturbation=self.feature_perturbation,
                    algorithm="linear",
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
        if self.space == "input":
            shap_values = X
            # Name columns
            if isinstance(X, pd.DataFrame):
                exp = X
            else:
                columns_name = ["Shap%d" % (i + 1) for i in range(X.shape[1])]

                exp = pd.DataFrame(
                    data=shap_values,
                    columns=columns_name,
                )
        if self.space == "prediction":
            try:
                shap_values = self.model.predict_proba(X)[:, 1]
            except:
                shap_values = self.model.predict(X)

            # Name columns
            exp = pd.DataFrame(
                data=shap_values,
                columns=["preds"],
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
        return roc_auc_score(self.Z_te, self.predict_proba(self.X_te)[:, 1])

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
            return self.gmodel.coef_[0]
        else:
            raise ValueError(
                "Detector model not supported. Supported models ar linear: {}, got {}".format(
                    self.supported_linear_detector, self.model.__class__.__name__
                )
            )
