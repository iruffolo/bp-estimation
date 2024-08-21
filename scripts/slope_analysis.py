import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plots(df):

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


if __name__ == "__main__":

    print("Analysing")

    path = "../data/results/slopes/"

    results = []

    for filename in os.listdir(path):
        if filename.endswith("csv"):
            print(filename)
            try:
                results.append(pd.read_csv(path + filename))
            except Exception as e:
                print(e)

    df = pd.concat(results, ignore_index=True)
    # df.to_csv(f"../data/results/slopes/combined.csv", index=False)

    plots(df)
