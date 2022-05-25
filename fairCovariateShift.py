# %%
import pandas as pd
from synthetic.fair_domain_adaptation_utils import gen_synth_shift_data

# %%
data_src, data_tar, sensible_feature, non_separating_feature = gen_synth_shift_data(
    gamma_shift_src=[0.0],
    gamma_shift_tar=[0.0],
    gamma_A=0.0,
    C_src=0,
    C_tar=1,
    N=1000,
    verbose=False,
)
# %%
