import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_slope(df):

    fig, ax = plt.subplots(2, figsize=(15, 10))

    ax[0].hist(df["slope"], bins=2000)
    ax[0].set_title("Corrected PAT vs SBP Slopes")
    ax[0].set_xlabel("Slope")
    ax[0].set_ylabel("Count")
    ax[0].set_xlim(-50, 50)

    ax[1].hist(df["naive_slope"], bins=2000)
    ax[1].set_title("Naive PAT vs SBP Slopes")
    ax[1].set_xlabel("Slope")
    ax[1].set_ylabel("Count")
    ax[1].set_xlim(-50, 50)

    plt.show()


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

    ax[1].plot(bins1[1:], cdf1, label="Corrected")
    ax[1].plot(bins2[1:], cdf2, label="Naive")
    ax[1].set_title(f"{metric} correlations CDF")
    ax[1].set_xlabel("Correlation")
    ax[1].set_ylabel("Probability")
    ax[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    print("Analysing")

    path = "../data/results/slopes/larger/"

    results = []

    for filename in os.listdir(path):
        if filename.endswith("csv"):
            try:
                data = pd.read_csv(path + filename)
                results.append(data)
                print(filename, len(data))
            except Exception as e:
                print(e)

    df = pd.concat(results, ignore_index=True)
    df.to_csv(f"../data/results/slopes/combined.csv", index=False)

    print(df.columns)

    # plot_slope(df)
    plot_correlation(df, "pearson")
    plot_correlation(df, "spearman")
