# %%
import numpy as np
import pandas as pd
from collections import namedtuple, Counter
import ot
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.model_selection import GridSearchCV
import seaborn as sns

import matplotlib.pyplot as plt
np.random.seed(101)
# %%
def random_logit(x):
    z = 1.0 / (1 + np.exp(-x))
    s = np.random.binomial(n=1, p=z)

    return 2 * s - 1


def gen_synth_shift_data(
    gamma_shift_src=[0.0],
    gamma_shift_tar=[0.0],
    gamma_A=0.0,
    C_src=0,
    C_tar=1,
    N=1000,
    verbose=False,
):
    """Flu classification example
    Variables
    C: context 0 or 1
    A: age group 0 or 1
    R: risk
    T: temperature
    Y: flu 0 or 1"""

    # Regression coefficients
    gamma_AC = 0.2
    gamma_TC = 0.2

    gamma_RA = -0.1
    gamma_YA = -0.8
    gamma_YR = 0.8
    gamma_TY = 0.8
    gamma_TR = 0.1
    gamma_TA = -0.8

    scale_T = 1.0
    scale_e = 0.64
    scale_A = 1.0
    scale_Y = 1.0

    # Source datasets
    data_src = []
    for gamma_shift in gamma_shift_src:
        C_src_vec = np.repeat(a=C_src, repeats=N)
        A = random_logit(
            scale_A
            * (
                gamma_A
                + gamma_shift * gamma_AC * C_src_vec
                + np.random.normal(loc=0.0, scale=scale_e, size=N)
            )
        )
        R = gamma_RA * A + np.random.normal(
            loc=0.0, scale=1 + scale_e, size=N
        )  # N(0,1)
        Y_src = random_logit(
            scale_Y
            * (
                gamma_YA * A
                + gamma_YR * R
                + np.random.normal(loc=0.0, scale=scale_e, size=N)
            )
        )
        T = (
            gamma_TY * Y_src
            + gamma_TR * R
            + gamma_TA * A
            + np.random.normal(
                loc=0.0, scale=scale_T + gamma_shift * gamma_TC * C_src_vec, size=N
            )
        )

        Y_src = (Y_src + 1) / 2
        X_src = np.stack([A, R, T], axis=1)
        data_src.append((gamma_shift, X_src, Y_src))

    # Target datasets
    data_tar = []
    for gamma_shift in gamma_shift_tar:
        C_tar_vec = np.repeat(a=C_tar, repeats=N)
        A = random_logit(
            scale_A
            * (
                gamma_A
                + gamma_shift * gamma_AC * C_tar_vec
                + np.random.normal(loc=0.0, scale=scale_e, size=N)
            )
        )
        R = gamma_RA * A + np.random.normal(loc=0.0, scale=1 + scale_e, size=N)
        Y_tar = random_logit(
            scale_Y
            * (
                gamma_YA * A
                + gamma_YR * R
                + np.random.normal(loc=0.0, scale=scale_e, size=N)
            )
        )
        T = (
            gamma_TY * Y_tar
            + gamma_TR * R
            + gamma_TA * A
            + np.random.normal(
                loc=0.0, scale=scale_T + gamma_shift * gamma_TC * C_tar_vec, size=N
            )
        )

        Y_tar = (Y_tar + 1) / 2
        X_tar = np.stack([A, R, T], axis=1)
        data_tar.append((gamma_shift, X_tar, Y_tar))

    sensible_feature = 0  # A
    non_separating_feature = 2  # T

    if verbose:
        print(
            "Coefficients: g_shift:{},gA:{},gAC:{},gRA:{},gYA:{},gYR:{},gTC:{},gTY:{},gTR:{},gTA:{}".format(
                gamma_shift,
                gamma_A,
                gamma_AC,
                gamma_RA,
                gamma_YA,
                gamma_YR,
                gamma_TC,
                gamma_TY,
                gamma_TR,
                gamma_TA,
            )
        )
    return data_src, data_tar, sensible_feature, non_separating_feature


def load_synth(data, target, data_test, target_test, smaller=False, scaler=True):
    len_train = len(data[:, -1])
    if scaler:
        scaler = StandardScaler()
        scaler.fit(data)
        data = scaler.transform(data)
        data_test = scaler.transform(data_test)
    if smaller:
        print("A smaller version of the dataset is loaded...")
        data = namedtuple("_", "data, target")(
            data[: len_train // 20, :-1], target[: len_train // 20]
        )
        data_test = namedtuple("_", "data, target")(data_test, target_test)
    else:
        data = namedtuple("_", "data, target")(data, target)
        data_test = namedtuple("_", "data, target")(data_test, target_test)
    return data, data_test



# %%
data_src, data_tar, sensible_feature, non_separating_feature= gen_synth_shift_data()

# %%
df = pd.DataFrame(data_src[0][1])
df.columns = ["Var%d" % (i + 1) for i in range(df.shape[1])]
df['target'] = data_src[0][2]
# %%
df['Var1'].unique()
# %%

# %%
sns.kdeplot(df.Var3)
# %%
