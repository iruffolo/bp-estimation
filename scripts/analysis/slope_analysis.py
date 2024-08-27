import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def add_grid(ax, min, max, step):

    # Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(min, max + 1, step)
    minor_ticks = np.arange(min, max + 1, step / 4)

    ax.set_xlim(min, max)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    # ax.set_yticks(major_ticks)
    # ax.set_yticks(minor_ticks, minor=True)

    # And a corresponding grid
    # ax.grid()

    # Or if you want different settings for the grids:
    ax.grid(which="minor", alpha=0.2)
    ax.grid(which="major", alpha=0.9)


def plot_slope(df, metric):

    fig, ax = plt.subplots(2, figsize=(15, 10))

    bins = int(len(df[metric]) / 20)

    thresh = 5000

    ax[0].hist(df[metric][abs(df[metric]) < thresh], bins=bins)
    ax[0].set_title("Corrected PAT vs SBP Slopes")
    ax[0].set_xlabel("Slope")
    ax[0].set_ylabel("Count")
    add_grid(ax[0], -4000, 4000, 1000)

    ax[1].hist(df[f"naive_{metric}"][abs(df[f"naive_{metric}"]) < thresh], bins=bins)
    ax[1].set_title("Naive PAT vs SBP Slopes")
    ax[1].set_xlabel("Slope")
    ax[1].set_ylabel("Count")
    add_grid(ax[1], -4000, 4000, 1000)

    # add_grid(ax[0], -4000, 2000)
    # add_grid(ax[1], -1000, 1000)

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_correlation(df, metric):

    fig, ax = plt.subplots(2, figsize=(15, 10))

    col1 = metric
    col2 = f"naive_{metric}"

    c1, bins1 = np.histogram(df[col1].dropna(), bins=100)
    c2, bins2 = np.histogram(df[col2].dropna(), bins=100)

    pdf1 = c1 / sum(c1)
    pdf2 = c2 / sum(c2)

    cdf1 = np.cumsum(pdf1)
    cdf2 = np.cumsum(pdf2)

    ax[0].plot(bins1[1:], pdf1, label="Corrected")
    ax[0].plot(bins2[1:], pdf2, label="Naive")
    ax[0].set_title(f"{metric} correlations PDF")
    ax[0].set_xlabel("Correlation")
    ax[0].set_ylabel("Probability")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(bins1[1:], cdf1, label="Corrected")
    ax[1].plot(bins2[1:], cdf2, label="Naive")
    ax[1].set_title(f"{metric} correlations CDF")
    ax[1].set_xlabel("Correlation")
    ax[1].set_ylabel("Probability")
    ax[1].legend()
    ax[1].grid()

    plt.tight_layout()
    plt.show()
    plt.close()


def combine_results(path, write=True):

    results = []

    for filename in os.listdir(path):
        if filename.endswith("csv") and filename != "combined.csv":
            try:
                data = pd.read_csv(path + filename)
                results.append(data)
                print(filename, len(data))
            except Exception as e:
                print(e)

    df = pd.concat(results, ignore_index=True)

    if write:
        df.to_csv(f"../data/results/slopes/combined.csv", index=False)

    return df


def check_stats(path):

    stats = {
        "successful": 0,
        "unexpected_failed": 0,
        "failed_alignment": 0,
        "poor_ecg_quality": 0,
        "poor_ppg_quality": 0,
    }

    for filename in os.listdir(path):
        if filename.endswith("npy"):

            data = np.load(path + filename, allow_pickle=True).item()

            stats["successful"] += data["successful"]
            stats["unexpected_failed"] += data["unexpected_failed"]
            stats["failed_alignment"] += data["failed_alignment"]
            stats["poor_ecg_quality"] += data["poor_ecg_quality"]
            stats["poor_ppg_quality"] += data["poor_ppg_quality"]

    print(stats)


if __name__ == "__main__":

    print("Analysing")

    path = "../data/results/median_slopes/"

    df = combine_results(path, False)
    print(len(df))

    # check_stats(path)
    # plot_slope(df, "median_slope")
    plot_correlation(df[:1000], "pearson")
    # plot_correlation(df, "spearman")
