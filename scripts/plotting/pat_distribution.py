import glob
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __name__ == "__main__":

    path = "/home/ian/dev/bp-estimation/data/paper_results"
    files = os.listdir(path)

    # for f in glob.glob(f"{path}/*[0-9]_pats.csv"):
    pats = []
    naives = []
    summaries = []

    devices = list(range(74, 116))

    for d in devices:

        pats_fn = [f for f in files if f"{d}_pats" in f]
        naive_fn = [f for f in files if f"{d}_naive" in f]
        summary_fn = [f for f in files if f"{d}_summary" in f]

        if pats_fn and naive_fn and summary_fn:
            pat = pd.read_csv(f"{path}/{pats_fn[0]}")
            naive = pd.read_csv(f"{path}/{naive_fn[0]}")
            summary = pd.read_csv(f"{path}/{summary_fn[0]}")

            pat["start_date"] = (pat["ecg_peaks"]).apply(datetime.fromtimestamp)
            naive["start_date"] = (naive["ecg_peaks"]).apply(datetime.fromtimestamp)

            pats.append(pat)
            naives.append(naive)
            summaries.append(summary)

            # print(pat)

        df = pd.concat(pats, ignore_index=True)

    print(len(df["corrected_pat"]))

    sns.kdeplot(df[df["start_date"] < "2017-01-03"], x="corrected_pat")
    sns.kdeplot(df[df["start_date"] >= "2017-01-03"], x="corrected_pat")
    plt.show()
