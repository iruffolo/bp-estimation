import matplotlib.pyplot as plt
import numpy as np


def plot_hist(h1, h2, bincenters, title="PATs", xlabel="", combine=False):

    fig, ax = plt.subplots()

    # Normalize first
    if combine:
        h = h1 + h2
        h = h / np.sum(h)
        ax.step(bincenters, h, where="mid", color="b")

    else:
        h1 = h1 / np.sum(h1)
        h2 = h2 / np.sum(h2)
        ax.step(bincenters, h1, where="mid", color="b")
        ax.step(bincenters, h2, where="mid", color="g")
        plt.legend(["Pre 2022", "Post 2022"])

    ax.minorticks_on()
    ax.yaxis.set_tick_params(which="minor", bottom=False)

    plt.title(f"{title}")
    plt.xlabel(xlabel)
    plt.ylabel("Probability")
    plt.grid(color="0.9")
    plt.savefig(f"{title.split()[0]}.svg", format="svg")

    plt.show()
    plt.close()
