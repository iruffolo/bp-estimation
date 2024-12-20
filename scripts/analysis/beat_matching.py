import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyhrv.nonlinear import poincare
from visualization.new_st import create_sawtooth, fit_sawtooth_phase

age_bins_day = np.linspace(0, 30, 31)
age_bins_month = [
    1,  # 1-3 months
    3,  # 3-6 months
    6,  # 6-9 months
    9,  # 9-12 months
    12,  # 12-18 months
    18,  # 18-24 months
    24,  # 2-3 years
    36,  # 3-4 years
    48,  # 4-6 years
    72,  # 6-8 years
    96,  # 8-12 years
    144,  # 12-15 years
    180,  # 15-18 years
    216,  # 18+
]


def find_bin(array, value):
    array = np.asarray(array)
    return array[np.where(array <= value)[0]][-1]


def age_in_months(d1, d2):

    d1 = datetime.fromtimestamp(d1)
    return (d1.year - d2.year) * 12 + d1.month - d2.month


def age_in_days(visit_time, dob):
    """
    Get age from patient ID

    :param pid: Patient ID
    :return: Age
    """

    age_at_visit = (datetime.fromtimestamp(visit_time) - dob).days

    return age_at_visit  # , age_at_visit / 365.2425


def calc_hist_stats(h, bins):

    mids = 0.5 * (bins[1:] + bins[:-1])
    avg = np.average(mids, weights=h)
    var = np.average((mids - avg) ** 2, weights=h)
    std = np.sqrt(var)

    return avg, std


def calc_sawtooth_one(times, pats):

    # Shift to zero for easy plotting
    x = times.values - times.iloc[0]
    y = pats.values

    x = x[0:2000]
    y = y[0:2000]

    period = 60
    amp = 25 / 1000

    windows = (x[-1] - x[0]) / (60 * 60)
    print(windows)

    st, fitp = fit_sawtooth_phase(x, y, period, amp)
    print(fitp)

    # N = len(x)
    # a = np.convolve(x, 50, mode="valid")
    # print(a)
    # print(a.shape)

    dx = x[1] - x[0]
    dydx = np.gradient(y, dx)
    print(dydx)

    fixed_st = (y - st) + (fitp[1] - amp / 2)

    # Sawtooth y values for plotting
    poly = np.polyfit(x, y, deg=100)
    z = np.poly1d(poly)

    fig, ax = plt.subplots()

    ax.scatter(x, y, marker="x")
    ax.scatter(x, st, marker=".", color="red")
    ax.scatter(x, fixed_st, alpha=0.5, marker=".")
    ax.plot(x, z(x), alpha=0.8, color="red")

    ax.scatter(x, dydx + fitp[1], color="green")

    print(f"orig: {np.mean(y)}")
    print(f"orig std: {np.std(y)}")
    print(f"st: {np.mean(fixed_st)}")
    print(f"st std: {np.std(fixed_st)}")

    plt.show()

    return y


def calc_hist(df, dobs):

    bins = 5000
    bin_range = (0, 5)

    _, edges = np.histogram([], bins=bins, range=bin_range)

    hists_day = {
        age: np.histogram([], bins=bins, range=bin_range)[0] for age in age_bins_day
    }
    hists_month = {
        age: np.histogram([], bins=bins, range=bin_range)[0] for age in age_bins_month
    }

    for idx, row in df.iterrows():
        age = row["age"]
        data = pd.read_csv(row["files"])

        dob = dobs[dobs["patient_id"] == row["pid"]]["dob"].values[0] / 10**9
        dob = datetime.fromtimestamp(dob)

        data["age_days"] = data["times"].apply(lambda x: age_in_days(x, dob))
        data["age_months"] = data["times"].apply(lambda x: age_in_months(x, dob))

        one_month = data[data["age_days"] <= 30]
        more_one_month = data[data["age_days"] > 30]
        # print(data.head())

        # # Data check
        _, sd1, sd2, sd_ratio, _ = poincare(rpeaks=data["times"], show=False)
        plt.close()
        if sd1 < 10 or sd2 < 10:
            print("Low SD, skipping")
            continue

        df = data[data["valid_correction"] > 0].sort_values("times")

        st = calc_sawtooth_one(df["times"], df["corrected_bm_pat"])

        if len(one_month) > 0:
            for day in one_month["age_days"].unique():
                print(f"days: {day}")

                h = np.histogram(
                    df["corrected_bm_pat"],
                    bins=bins,
                    range=bin_range,
                )[0]

                hists_day[day] += h

        if len(more_one_month) > 0:

            for month in more_one_month["age_months"].unique():
                print(f"months: {month}")

                h = np.histogram(
                    df["corrected_bm_pat"],
                    bins=bins,
                    range=bin_range,
                )[0]

                bin = find_bin(age_bins_month, month)
                print(f"bin {bin}")

                hists_month[bin] += h

    for month, hist in hists_month.items():
        if np.sum(hist) > 0:
            avg, std = calc_hist_stats(hist, edges)
            print(f"{month}: {avg} +- {std}")

    return hists_day, hists_month

    # fig, ax = plt.subplots(1, figsize=(10, 10))
    # for age, hist in hists.items():
    #     # ax.step(bincenters, hist, where="mid", label=f"{age} years")
    #     # if age in [3, 6, 9, 12]:
    #     cumsum = np.cumsum(hist)
    #     cumsum = cumsum / cumsum[-1]
    #     ax.plot(edges[:-1], cumsum, label=f"{age} years")
    #     # hist = hist / np.sum(hist)
    #     # ax.stairs(hist, edges, label=f"{age} years", alpha=0.8)
    # ax.minorticks_on()
    # ax.yaxis.set_tick_params(which="minor", bottom=False)
    # ax.set_title(f"PAT cdf")
    # ax.set_xlabel("PAT (s)")
    # ax.set_ylabel("Probability")
    # ax.grid(color="0.9")
    # ax.legend(loc="upper right")
    # plt.show()


if __name__ == "__main__":

    # datapath = "/home/ian/dev/bp-estimation/data/beat_matching/"
    datapath = "/home/ian/dev/bp-estimation/data/beat_matching2/"

    dobpath = "/home/ian/dev/bp-estimation/data/ian_dataset_dobs.csv"
    dobs = pd.read_csv(dobpath)

    patients = {
        "pid": [],
        "dev": [],
        "age": [],
        "files": [],
        "date": [],
    }
    for file in os.listdir(datapath):
        if file.endswith("pat.csv"):
            # print(f"Processing {file}")
            _, dev, pid, month, year, age, ftype = file.split("_")

            patients["pid"].append(int(pid))
            patients["dev"].append(int(dev))
            patients["age"].append(int(age))
            patients["files"].append(datapath + file)
            patients["date"].append(datetime(year=int(year), month=int(month), day=1))

    df = pd.DataFrame(patients)
    pre = df[df["date"] <= "2022-06-01"]
    h1, h2 = calc_hist(pre, dobs)

    df1 = pd.DataFrame(h1)
    df2 = pd.DataFrame(h2)

    save_path = "/home/ian/dev/bp-estimation/data"
    fn1 = os.path.join(save_path, "daily_hists.csv")
    fn2 = os.path.join(save_path, "monthly_hists.csv")

    df1.to_csv(fn1, header="column_names", index=False)
    df2.to_csv(fn2, header="column_names", index=False)
