import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if __name__ == "__main__":

    path = "/home/ian/dev/bp-estimation/data/paper_results/"
    files = os.listdir(path)

    pats_fn = [f for f in files if f"_pats.csv" in f]
    pat = pd.read_csv(f"{path}/{pats_fn[0]}")

    pids = pat["patient_id"].unique()

    for p in pids:
        p1 = pat[pat["patient_id"] == p]
        sns.scatterplot(data=p1, x="ecg_peaks", y="corrected_pat", hue="patient_id")
        plt.show()
