import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_scatter(df, name, r1=None, r2=None):

    s = df["slope_ppm"]
    p = df["period"]

    # ax = sns.heatmap(df[["slope_ppm", "period"]], annot=True)

    fig, ax = plt.subplots()
    ax.hist2d(s, p, bins=1000, cmap="Reds")
    # ax.scatter(s, p, s=0.01, alpha=0.02)
    ax.set_xlim(r1)
    ax.set_ylim(r2)
    ax.set_xlabel("Slope (ppm)")
    ax.set_ylabel("Period (s)")
    ax.set_title(name)
    plt.tight_layout()
    plt.savefig(f"{name}.png")
    plt.close()


def plot_dists(df, name, bins=1000, r1=None, r2=None):

    s = df["slope_ppm"]
    p = df["period"]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].hist(s, bins=bins, range=r1)
    ax[0].set_xlabel("Slope (ppm)")
    ax[0].set_ylabel("Count")
    ax[0].set_title(f"Slopes: mean={np.mean(s):.3f}, std={np.std(s):.3f}")

    ax[1].hist(p, bins=bins, range=r2)
    ax[1].set_xlabel("Period (s)")
    ax[0].set_ylabel("Count")
    ax[1].set_title(f"Periods: mean={np.mean(p):.3f}, std={np.std(p):.3f}")

    plt.suptitle(name)
    plt.tight_layout()
    plt.savefig(f"{name}.png")
    plt.close()


if __name__ == "__main__":

    path = "/home/iruffolo/dev/bp-estimation/data/st_correction/"
    path = "/home/iruffolo/dev/bp-estimation/data/st_correction_post/"

    files = os.listdir(path)

    st1 = []
    st2 = []

    for f in files:
        df = pd.read_csv(path + f)

        if "st1" in f:
            st1.append(df)
        if "st2" in f:
            st2.append(df)

    st1 = pd.concat(st1)
    st2 = pd.concat(st2)

    print(st1["slope_ppm"].mean(), st1["slope_ppm"].std())
    print(st2["slope_ppm"].mean(), st2["slope_ppm"].std())

    print(len(st1), len(st2))

    fig, ax = plt.subplots()
    ax.hist(st2["points"], bins=200)
    plt.savefig("points_hist.png")
    plt.close()

    # Drop fits with low number of points
    st1 = st1[st1["points"] > 50]
    st2 = st2[(st2["points"] > 100) & (st2["points"] < 400) & (st2["slope_ppm"] > 0)]

    patients = st2["p_id"].unique()

    plot_dists(st1, "sawtooth1_post", r1=(0, 1000), r2=(0, 250))
    plot_dists(st2, "sawtooth2_post", r1=(-200, 600), r2=(0, 1000))
    plot_scatter(st2, "st2_scatter_post", r1=(0, 300), r2=(0, 300))
    plot_scatter(st1, "st1_scatter_post", r1=(0, 1000), r2=(0, 250))

    # print("Plotting individual patients")
    # for p in patients:
    #     plot_dists(st2[st2["p_id"] == p], f"plots/st2_{p}")
    #     plot_scatter(st2[st2["p_id"] == p], f"scatter_plots/st2_{p}")
