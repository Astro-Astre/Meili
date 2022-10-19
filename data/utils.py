import pandas as pd
import os
import numpy as np


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def load_dir(dir) -> pd.DataFrame:
    local_files = os.listdir(dir)
    ra, dec = [], []
    for i in range(len(local_files)):
        if ".fits" in local_files[i]:
            t_ra, t_dec = float(local_files[i].split("_")[0]), float(local_files[i].split("_")[1].split(".fits")[0])
            ra.append(t_ra)
            dec.append(t_dec)
    return pd.DataFrame(list(zip(ra, dec)), columns=["ra", "dec"])


def answer_prob(out):
    output = []
    q1 = out[0] + out[1] + out[2]
    q2 = out[3] + out[4]
    q3 = out[5] + out[6]
    q4 = out[7] + out[8] + out[9]
    q5 = out[10] + out[11] + out[12] + out[13] + out[14]
    q6 = out[15] + out[16] + out[17]
    q7 = out[18] + out[19] + out[20]
    q8 = out[21] + out[22] + out[23]
    q9 = out[24] + out[25] + out[26] + out[27] + out[28] + out[29]
    q10 = out[30] + out[31] + out[32] + out[33]

    output.append(
        [out[0] / q1, out[1] / q1, out[2] / q1, out[3] / q2, out[4] / q2, out[5] / q3, out[6] / q3, out[7] / q4,
         out[8] / q4,
         out[9] / q4, out[10] / q5, out[11] / q5, out[12] / q5, out[13] / q5, out[14] / q5, out[15] / q6, out[16] / q6,
         out[17] / q6, out[18] / q7, out[19] / q7, out[20] / q7, out[21] / q8, out[22] / q8, out[23] / q8, out[24] / q9,
         out[25] / q9, out[26] / q9, out[27] / q9, out[28] / q9, out[29] / q9, out[30] / q10, out[31] / q10,
         out[32] / q10, out[33] / q10])
    return np.array(output)
