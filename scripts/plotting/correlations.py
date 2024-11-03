import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate


def plot_hist(h1, h2, bincenters):

    fig, ax = plt.subplots()

    # Normalize first
    h1 = h1 / np.sum(h1)
    h2 = h2 / np.sum(h2)

    # ax.plot(bincenters, h1, ".", markersize=4, color="b")
    # bspline = interpolate.make_interp_spline(bincenters, h1, 2)
    # plt.plot(bincenters, bspline(bincenters))
    ax.step(bincenters, h1, where="mid", color="b")
    ax.step(bincenters, h2, where="mid", color="g")

    ax.minorticks_on()
    ax.yaxis.set_tick_params(which="minor", bottom=False)

    plt.title("Spearman Correlation PDF")
    plt.xlabel("Spearman Correlation Coefficient")
    plt.ylabel("Probability")
    plt.grid(color="0.9")
    plt.legend(["Corrected", "Naive"])

    plt.savefig(f"spearman_pdf.svg", format="svg")
    plt.show()
    plt.close()


def plot_cdf(h1, h2, edges):

    # Calculate cumulative sums for CDF
    cusum1 = np.cumsum(h1)
    cusum1 = cusum1 / cusum1[-1]
    cusum2 = np.cumsum(h2)
    cusum2 = cusum2 / cusum2[-1]

    plt.plot(edges[:-1], cusum1, color="b")
    plt.plot(edges[:-1], cusum2, color="g")

    plt.title("Spearman Correlation CDF")
    plt.xlabel("Spearman Correlation Coefficient")
    plt.ylabel("Probability")
    plt.grid(color="0.9")
    plt.legend(["Corrected", "Naive"])

    plt.savefig(f"spearman_cdf.svg", format="svg")
    plt.show()
    plt.close()


if __name__ == "__main__":

    path = "/home/ian/dev/bp-estimation/data/paper_results/"
    files = os.listdir(path)
    summary_files = [f for f in files if f"summary" in f]

    bins = 500
    bin_range = (-1, 1)

    h1, edges = np.histogram([], bins=bins, range=bin_range)
    h2 = np.histogram([], bins=bins, range=bin_range)[0]
    bincenters = [(edges[i] + edges[i + 1]) / 2.0 for i in range(len(edges) - 1)]

    for f in summary_files:

        print(f"Processing file {f}")

        summary = pd.read_csv(f"{path}/{f}")
        print(summary)

        h1 += np.histogram(summary["spearman"], bins=bins, range=bin_range)[0]
        h2 += np.histogram(summary["naive_spearman"], bins=bins, range=bin_range)[0]

    plot_hist(h1, h2, bincenters)
    plot_cdf(h1, h2, edges)
