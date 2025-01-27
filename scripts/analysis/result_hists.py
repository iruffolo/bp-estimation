import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Bins defined for grouping histograms
age_bins = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,  # Up to 30 days
    90,  # 3 Months
    180,  # 6 Months
    270,  # 9 Months
    365,  # 12 Months
    365 + 180,  # 18 Months
    365 * 2,  # 2 Years
    365 * 3,  # 3 Years
    365 * 4,  # 4 Years
    365 * 6,  # 6 Years
    365 * 8,  # 8 Years
    365 * 12,  # 12 Years
    365 * 15,  # 15 Years
    365 * 18,  # 18 Years
]


def calc_hist_stats(h, bins):

    mids = 0.5 * (bins[1:] + bins[:-1])
    avg = np.average(mids, weights=h)
    var = np.average((mids - avg) ** 2, weights=h)
    std = np.sqrt(var)

    return avg, std


def combine(path):

    files = os.listdir(path)
    pkls = [f for f in files if ".pkl" in f]

    df = pd.read_pickle(path + pkls[0])

    combined = {
        "naive": np.array(df["naive"].tolist()),
        "bm": np.array(df["bm"].tolist()),
        "bm_st1": np.array(df["bm_st1"].tolist()),
        "bm_st1_st2": np.array(df["bm_st1_st2"].tolist()),
    }

    for f in pkls[1:]:
        df = pd.read_pickle(path + f)

        for key in df:
            combined[key] += np.array(df[key].tolist())

    for key in combined:
        np.save(path + key, combined[key])


if __name__ == "__main__":

    print("Processing results")

    path = "../../data/result_histograms_prod/"
    files = os.listdir(path)

    names = ["naive", "bm", "bm_st1", "bm_st1_st2"]

    # Define hist edges for plotting
    _, edges = np.histogram([], bins=5000, range=(0, 5))

    cols = [n for n in names]
    cols += ["bm_st1_st2_offset"]
    stats = {c: {"avg": [], "median": [], "std": []} for c in cols}
    stats["title"] = []

    offset = 1.31

    for n in names:
        print(f"Processing {n}")
        df = np.load(path + n + ".npy")

        for i in range(len(age_bins) - 1):

            hist = df[age_bins[i] : age_bins[i + 1]]

            title = f"Age {age_bins[i]}:{age_bins[i+1]} days"
            if np.sum(hist) == 0:
                print(f"No data for {title}")
                continue

            hist = np.sum(hist, axis=0)

            # Normalize for plot
            n_hist = hist / np.sum(hist)
            mcumsum = np.cumsum(n_hist)
            idx = np.where(mcumsum < 0.5)[0][-1]

            avg, std = calc_hist_stats(hist.flatten(), edges)

            stats[n]["avg"].append(avg)
            stats[n]["std"].append(std)
            stats[n]["median"].append(idx / 1000)

            if n == "bm_st1_st2":
                stats["bm_st1_st2_offset"]["avg"].append(avg - offset)
                stats["bm_st1_st2_offset"]["std"].append(std)
                stats["bm_st1_st2_offset"]["median"].append(idx / 1000 - offset)

            if n == "naive":
                stats["title"].append(title)

            continue

            fig, ax = plt.subplots(1, 2)
            # ax[0].bar(
            #     edges[:-1], hist, width=np.diff(edges), align="edge", edgecolor="black"
            # )
            ax[0].step(
                edges,
                np.append(hist, 0),
                where="post",
                linewidth=2,
                label=title,
            )

            cumsum = np.cumsum(hist)
            cumsum = cumsum / cumsum[-1]
            ax[1].plot(edges[:-1], cumsum, label=title)

            ax[0].minorticks_on()
            ax[0].yaxis.set_tick_params(which="minor", bottom=False)
            ax[1].minorticks_on()
            ax[1].yaxis.set_tick_params(which="minor", bottom=False)

            plt.xlabel("PAT")
            plt.xlim([0.5, 2])
            plt.ylabel("Probability")
            plt.grid(color="0.9")
            plt.legend(loc="upper right")
            plt.tight_layout()
            plt.savefig(f"result_hists/{n}/{age_bins[i]}.png")
            plt.savefig(f"result_hists/{n}/{age_bins[i]}.svg", format="svg")
            # plt.show()
            plt.close()

    fig, ax = plt.subplots(5, 2, figsize=(10, 10))

    x1 = age_bins[3:30]
    x2 = age_bins[30:]

    names.append("bm_st1_st2_offset")

    size = 10

    for i, n in enumerate(names):

        df = pd.DataFrame(stats[n])

        ax[i][0].scatter(x1, df["avg"][0:27], s=size, marker="x")
        ax[i][0].scatter(x1, df["median"][0:27], s=size, marker="x")
        ax[i][0].errorbar(x1, df["avg"][0:27], yerr=df["std"][0:27], fmt=".")

        poly = np.polyfit(x1, df["avg"][0:27], 1)
        poly1d = np.poly1d(poly)
        ax[i][0].plot(x1, poly1d(x1), color="green")

        # poly = np.polyfit(x1, df["median"][0:27], 1)
        # poly1d = np.poly1d(poly)
        # ax[i][0].plot(x1, poly1d(x1), color="red")

        ax[i][0].set_ylabel("PAT (s)")

        ax[i][1].scatter(x2, df["avg"][27:], label="avg", s=size)
        ax[i][1].scatter(x2, df["median"][27:], label="median", s=size, marker="x")
        ax[i][1].errorbar(x2, df["avg"][27:], yerr=df["std"][27:], fmt=".")

        poly = np.polyfit(x2, df["avg"][27:], 1)
        poly1d = np.poly1d(poly)
        ax[i][1].plot(x2, poly1d(x2), color="green")

        # poly = np.polyfit(x2, df["median"][27:], 1)
        # poly1d = np.poly1d(poly)
        # ax[i][1].plot(x2, poly1d(x2), color="red")

        ax[i][1].set_xlim((20, x2[-1] + 10))

        # ax[i][0].set_ylim((0, 2))
        ax[i][1].sharey(ax[i][0])

        ax[i][1].legend(loc="upper right")

        ax[i][0].grid()
        ax[i][1].grid()

        # ax[i][1].set_yticks([])

    ax[0][0].set_title(f"Naive")
    ax[1][0].set_title(f"Beatmatching Only")
    ax[2][0].set_title(f"Beatmatching with sawtooth 1 corrected")
    ax[3][0].set_title(f"Beatmatching with sawtooth 1 and 2 corrected")
    ax[4][0].set_title(f"Beatmatching with sawtooth 1 and 2 corrected, offset applied")

    plt.suptitle("PAT by Age")
    plt.xlabel("Age (days)")
    plt.tight_layout()
    plt.savefig("pat_by_age.svg", format="svg")
    plt.show()
