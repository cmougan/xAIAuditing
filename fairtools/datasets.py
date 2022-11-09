import pandas as pd
from sklearn.datasets import make_blobs
from folktables import ACSDataSource, ACSTravelTime
import numpy as np

d = {
    "AGEP": "Age",
    "SCHL": "Education",
    "MAR": "Marital",
    "ESP": "Employment",
    "ST": "State",
    "POVPIP": "PovertyIncome",
    "MIG": "MobilityStat",
    "CIT": "Citizenship",
    "DIS": "Disability",
    "OCCP": "Occupation",
    "PUMA": "Area",
    "JWTR": "WorkTravel",
    "JWTRNS": "WorkTravel2",
    "RAC1P": "Race",
    "AGEP": "Age",
    "POWPUMA": "WorkPlace",
}


class GetData:
    """
    Example:
    from tools.datasets import GetData
    data = GetData(type="blobs")
    X, y, X_ood, y_ood = data.get_data()
    """

    def __init__(self, type: str = "blobs", N: int = 100000):
        self.type = type
        self.N = N
        self.X = None
        self.y = None
        self.X_tr = None
        self.X_te = None
        self.y_tr = None
        self.y_te = None
        self.X_ood = None
        self.y_ood = None
        self.supported_types = ["blobs", "synthetic", "real"]
        assert self.type in self.supported_types

    def get_data(self):

        ## Real data based on US census data
        data_source = ACSDataSource(
            survey_year="2014", horizon="1-Year", survey="person"
        )
        try:
            acs_data = data_source.get_data(states=["CA"], download=False)
        except:
            acs_data = data_source.get_data(states=["CA"], download=True)
        X, y, group = ACSTravelTime.df_to_numpy(acs_data)
        X = pd.DataFrame(X, columns=ACSTravelTime.features).rename(columns=d)

        self.X = X.rename(columns=d)

        # Lets make smaller data for computational reasons
        self.X = X.head(self.N)
        self.y = y[: self.N]
        self.group = group[: self.N]

        return self.X, self.y, self.group

    def get_state(self, year: str = "2014", state: str = "NY"):
        # OOD data
        data_source = ACSDataSource(survey_year=year, horizon="1-Year", survey="person")
        try:
            acs_data = data_source.get_data(states=[state], download=False)
        except:
            acs_data = data_source.get_data(states=[state], download=True)
        ca_features, ca_labels, ca_group = ACSTravelTime.df_to_numpy(acs_data)
        ca_features = pd.DataFrame(ca_features, columns=ACSTravelTime.features).rename(
            columns=d
        )

        # Filter to only have groups 1 and 2
        ca_features["group"] = ca_group
        ca_features["label"] = ca_labels
        ca_features = ca_features[
            (ca_features["group"] == 8) | (ca_features["group"] == 6)
        ]
        ca_features["group"] = np.where(ca_features["group"] == 8, 1, 0)
        # ca_features["group"] = ca_features["group"].values - 1  # This is to make it 0 and 1

        # Split data
        X = ca_features.drop(["label", "RAC1P"], axis=1)
        X_ = X.drop(["group"], axis=1)
        y = ca_features["group"]

        # Shorten data
        self.X = X.head(self.N)
        self.y = y[: self.N]

        return self.X, self.y
