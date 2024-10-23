import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __name__ == "__main__":

    devices = list(range(74, 116))

    path = "/home/ian/dev/bp-estimation/data/paper_results"
    files = os.listdir(path)

    # for f in glob.glob(f"{path}/*[0-9]_pats.csv"):
    # print(f)

    # for f in files:
    # print(f)

    pats = []
    naives = []
    summaries = []

    for d in devices:

        pats_fn = [f for f in files if f"{d}_pats" in f]
        naive_fn = [f for f in files if f"{d}_naive" in f]
        summary_fn = [f for f in files if f"{d}_summary" in f]

        if pats_fn and naive_fn and summary_fn:
            pat = pd.read_csv(f"{path}/{pats_fn[0]}")
            naive = pd.read_csv(f"{path}/{naive_fn[0]}")
            summary = pd.read_csv(f"{path}/{summary_fn[0]}")

            pats.append(pat)

            # print(len(pats))
            # print(pats)
            # print(summary)

        df = pd.concat(pats, ignore_index=True)

    sns.displot(df["corrected_pat"], bins=100, kde=True)
    plt.show()
