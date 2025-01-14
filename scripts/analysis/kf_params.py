import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

    print(len(st1), len(st2))

    # Drop fits with low number of points
    st1 = st1[st1["points"] > 50]
    st2 = st2[st2["points"] > 80]

    plot_dists(st1, "sawtooth1", r1=(0, 1000), r2=(0, 250))
    plot_dists(st2, "sawtooth2", r1=(-200, 600), r2=(0, 1000))
