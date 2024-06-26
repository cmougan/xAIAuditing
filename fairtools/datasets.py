import pandas as pd
from sklearn.datasets import make_blobs
from folktables import (
    ACSDataSource,
    ACSTravelTime,
    ACSIncome,
    ACSEmployment,
    ACSMobility,
    ACSPublicCoverage,
)
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
    "SEX": "Sex",
    "RELP": "Relationship",
    "POBP": "PlaceOfBirth",
    "ANC": "Ancestry",
    "MIL": "Military",
    "DEYE": "VisionDiff",
    "DEAR": "EaringDiff",
    "DREAM": "CognitiveDiff",
    "ESR": "EmploymentStatus",
    "WKHP": "WorkedHours",
    "COW": "ClassOfWorker",
}

r = {
    1: "White alone",
    2: "Black or African American alone",
    3: "American Indian",
    4: "Alaska Native",
    5: "American Indian",
    6: "Asian Indian",
    7: "Hawaiian",
    8: "Other",
    9: "Mixed Race",
}


class GetData:
    """
    Example:
    from tools.datasets import GetData
    data = GetData(type="blobs")
    X, y, X_ood, y_ood = data.get_data()
    """

    def __init__(self, type: str = "blobs", N: int = 20_000):
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

    def get_state(
        self,
        year: str = "2014",
        state: str = "NY",
        datasets: str = "ACSIncome",
        group1: int = 6,
        group2: int = 8,
        verbose: str = False,
    ):
        if datasets == "ACSTravelTime":
            self.dataset = ACSTravelTime
        elif datasets == "ACSEmployment":
            self.dataset = ACSEmployment
        elif datasets == "ACSIncome":
            self.dataset = ACSIncome
        elif datasets == "ACSMobility":
            self.dataset = ACSMobility
        elif datasets == "ACSPublicCoverage":
            self.dataset = ACSPublicCoverage

        data_source = ACSDataSource(survey_year=year, horizon="1-Year", survey="person")
        try:
            acs_data = data_source.get_data(states=[state], download=False)
        except:
            acs_data = data_source.get_data(states=[state], download=True)

        ca_features, ca_labels, ca_group = self.dataset.df_to_numpy(acs_data)
        ca_features = pd.DataFrame(ca_features, columns=self.dataset.features).rename(
            columns=d
        )

        # Filter to only have groups 1 and 2
        ca_features["group"] = ca_group
        ca_features["label"] = ca_labels
        if verbose:
            print(ca_features["group"].value_counts())
        ca_features = ca_features[
            (ca_features["group"] == group1) | (ca_features["group"] == group2)
        ]
        # Remove some feats
        try:
            ca_features = ca_features.drop(columns=["PlaceOfBirth"])
        except:
            print("Couldnt remove PlaceOfBirth")
        # Rename features
        ca_features["group"] = np.where(ca_features["group"] == group1, 1, 0)
        # ca_features["group"] = ca_features["group"].values - 1  # This is to make it 0 and 1

        # Split data
        self.X = ca_features.drop(["label", "Race"], axis=1)
        self.y = ca_features["label"]

        # Shorten data
        self.X = self.X.head(self.N)
        self.y = self.y[: self.N]

        return self.X, self.y
