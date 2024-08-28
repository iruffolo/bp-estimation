import matplotlib.pyplot as plt
import numpy as np


def plot_slopes(synced, f1, f2, l1, l2):
    """
    Plot the slopes of the lines of best fit for the corrected and naive PATs
    """

    # Get data from lines of best fit for plotting
    xx1, yy1 = l1.linspace(domain=[0, 2])
    xx2, yy2 = l2.linspace(domain=[0, 2])

    fig, ax = plt.subplots(2, figsize=(15, 10))

    lw = 3
    ms = 5

    ax[0].plot(synced["pats"], synced["bp"], ".", alpha=0.5)
    ax[0].plot(f1["pats"]["median"], f1["bp"], "ro", markersize=ms, label="Medians")
    ax[0].plot(xx1, yy1, label=f"{l1.convert().coef}", linewidth=lw)
    ax[0].set_title(f"Corrected PATs")
    ax[0].legend(loc="upper left")

    ax[1].plot(synced["naive_pats"], synced["bp"], ".", alpha=0.5)
    ax[1].plot(
        f2["naive_pats"]["median"], f2["bp"], "ro", markersize=ms, label="Medians"
    )
    ax[1].plot(xx2, yy2, label=f"{l2.convert().coef}", linewidth=lw)
    ax[1].set_title(f"Naive PATs")
    ax[1].legend(loc="upper right")

    for a in ax:
        a.set_xlim(0, 2)
        a.set_ylim(min(synced["bp"] - 5), max(synced["bp"]) + 5)
        a.set_xlabel("PAT (s)")
        a.set_ylabel("BP (mmHG)")

    plt.tight_layout()
    plt.show()

    plt.close()
