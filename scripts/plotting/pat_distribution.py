import glob
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from distribution_plots import plot_hist
from matplotlib.ticker import MultipleLocator
from scipy.stats import norm

if __name__ == "__main__":

    path = "/home/ian/dev/bp-estimation/data/paper_results/"
    files = os.listdir(path)

    devices = list(range(74, 116))

    bins = 1000
    bin_range = (0, 3)

    H1, edges = np.histogram([], bins=bins, range=bin_range)
    H2, _ = np.histogram([], bins=bins, range=bin_range)
    H3, _ = np.histogram([], bins=bins, range=bin_range)
    H4, _ = np.histogram([], bins=bins, range=bin_range)
    bincenters = [(edges[i] + edges[i + 1]) / 2.0 for i in range(len(edges) - 1)]

    for d in devices:

        print(f"Processing device {d}")

        pats_fn = [f for f in files if f"{d}_pats" in f]
        naive_fn = [f for f in files if f"{d}_naive" in f]

        if pats_fn and naive_fn:
            pat = pd.read_csv(f"{path}/{pats_fn[0]}")
            naive = pd.read_csv(f"{path}/{naive_fn[0]}")

            pat["start_date"] = (pat["ecg_peaks"]).apply(datetime.fromtimestamp)
            naive["start_date"] = (naive["ecg_peaks"]).apply(datetime.fromtimestamp)

            pre = pat[pat["start_date"] < "2022-02-01"]
            post = pat[pat["start_date"] > "2022-12-01"]
            H1 += np.histogram(pre["corrected_pat"], bins=bins, range=bin_range)[0]
            H2 += np.histogram(post["corrected_pat"], bins=bins, range=bin_range)[0]

            pre = naive[naive["start_date"] < "2022-06-01"]
            post = naive[naive["start_date"] > "2022-09-01"]
            H3 += np.histogram(pre["naive_pat"], bins=bins, range=bin_range)[0]
            H4 += np.histogram(post["naive_pat"], bins=bins, range=bin_range)[0]

    plot_hist(H1, H2, bincenters, "Corrected", combine=True)
    plot_hist(H3, H4, bincenters, "Naive", combine=True)
