import glob
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import seaborn as sns
from atriumdb import AtriumSDK


def plot_hist(files, save=True, show=True):
    """
    Plot histogram of PAT data

    :param data: PAT data
    :param p_id: Patient ID
    :param save: Save plot
    :return: None
    """

    slice_l = 1250
    slice_u = 2500

    bins = np.linspace(0, 4, 5000)[slice_l : slice_u + 1]

    for r in files:

        data = np.load(f"{r}", allow_pickle=True).item()

        for p_id, data in data.items():
            print(p_id, data)

            fig, ax = plt.subplots(1, figsize=(10, 10))

            ax.stairs(data[slice_l:slice_u], bins, fill=True, alpha=0.8)

            plt.suptitle(f"PAT Distribution for Patient {p_id}")
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency")

            if save:
                plt.savefig(f"plots/histograms/pat_hist_{p_id}.png")

            if show:
                plt.show()

            plt.close()


def get_age_at_visit(visit_time, dob):
    """
    Get age from patient ID

    :param pid: Patient ID
    :return: Age
    """

    age_at_visit = (datetime.fromtimestamp(visit_time) - dob).days

    return age_at_visit  # , age_at_visit / 365.2425


def plot_by_date(sdk, path, files, shape=4999):
    """
    Split plot into date ranges
    """

    total_pats = {}

    for f in files:
        dev, pid = f.strip(".csv").split("_")

        dob = datetime.fromtimestamp(sdk.get_patient_info(int(pid))["dob"] / 10**9)
        print(dob)
        # eastern_tz = pytz.timezone("US/Eastern")
        # print(eastern_tz.localize(dob))

        data = pd.read_csv(f"{path}/{f}")
        print(data.head())

        data["age_days"] = data["times"].apply(lambda x: get_age_at_visit(x, dob))
        data["age_years"] = data["age_days"] / 365.2425

        year = pd.to_datetime(data["times"], unit="s").dt.year
        print(data.head())

    return

    for k, v in data.items():
        print(k, v)

        year = pd.to_datetime(v["visit_time"], unit="s").year
        month = pd.to_datetime(v["visit_time"], unit="s").month
        print(year, month)

        if year not in total_pats:
            total_pats[year] = {}
        if month not in total_pats[year]:
            total_pats[year][month] = np.zeros(shape)

        total_pats[year][month] += v["pat"]
        # print(np.sum(total_pats))

    slice_l = 500
    slice_u = 3500
    bins = np.linspace(0, 4, 5000)[slice_l : slice_u + 1]

    fig, ax = plt.subplots(12, figsize=(10, 10))

    for m in total_pats[2022].keys():
        ax[m - 1].stairs(
            total_pats[2022][m][slice_l:slice_u],
            bins,
            fill=True,
            alpha=0.8,
            label=f"{m}",
        )
        ax[m - 1].legend()
    # for year in total_pats.keys():
    #     ax.stairs(total_pats[year][slice_l:slice_u], bins, fill=True, alpha=0.8,
    #               label=f"{year}")

    plt.legend()
    plt.suptitle(f"PAT Distributions by Year")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency")

    plt.show()


def compile_plots(files, shape=4999):
    """
    Compile all plots into a single density plot

    :return: None
    """

    total_pats = np.zeros(shape)
    print(total_pats.shape)

    for r in files:

        data = np.load(f"{r}", allow_pickle=True).item()

        for k, v in data.items():
            print(k, v)
            total_pats += v["pat"]
            print(np.sum(total_pats))

    slice_l = 500
    slice_u = 3500
    bins = np.linspace(0, 4, 5000)[slice_l : slice_u + 1]

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.stairs(total_pats[slice_l:slice_u], bins, fill=True, alpha=0.8)

    plt.suptitle(f"Total PAT Distribution ({np.sum(total_pats)} measurements)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency")

    plt.show()


def get_age_from_dob(dob):
    """
    Get age from date of birth

    :param dob: Date of birth (nanosecond epoch)
    :return: Age
    """

    return (datetime.now() - datetime.fromtimestamp(dob / 10**9)).days / 365.2425


def plot_by_age(files, shape=4999, max_age=15):
    """
    Map ages to patient IDs

    :return: None
    """

    local_dataset = "/mnt/datasets/atriumdb_abp_estimation_2024_02_05"
    sdk = AtriumSDK(dataset_location=local_dataset)

    pat_by_age = {}

    for r in files:

        data = np.load(f"{r}", allow_pickle=True).item()

        for pat_id, v in data.items():
            if pat_id:
                info = sdk.get_patient_info(pat_id)
                age = min(int(get_age_from_dob(info["dob"])), max_age)

                print(f"Patient {pat_id} is {age} years old")
                print(np.sum(v["pat"]))

                if age not in pat_by_age:
                    pat_by_age[age] = {}
                    pat_by_age[age]["data"] = np.zeros(shape)
                    pat_by_age[age]["count"] = 0

                pat_by_age[age]["data"] += v["pat"]
                pat_by_age[age]["count"] += 1

    print(pat_by_age)
    pat_by_age = dict(sorted(pat_by_age.items()))

    fig, ax = plt.subplots(len(pat_by_age), figsize=(10, 35))

    slice_l = 1250
    slice_u = 2500
    bins = np.linspace(0, 4, 5000)[slice_l : slice_u + 1]

    for i, (age, pat) in enumerate(pat_by_age.items()):
        ax[i].stairs(pat["data"][slice_l:slice_u], bins, fill=True, alpha=0.8)
        ax[i].set_title(
            f"PAT Distribution for Patients {age} years old ({pat['count']} patients)"
        )
        ax[i].set_xlabel("Time (s)")
        ax[i].set_ylabel("Frequency")
        ax[i].sharex(ax[0])

    plt.tight_layout()
    plt.savefig("plots/pat_by_age_shuffle.png")
    # plt.show()


if __name__ == "__main__":

    path = "/home/ian/dev/bp-estimation/data/pats"
    files = os.listdir(path)
    print(f"Files: {files}")

    # Mounted dataset
    local_dataset = "/home/ian/dev/datasets/ian_dataset_2024_08_26"
    sdk = AtriumSDK(dataset_location=local_dataset)

    # print(sdk.get_all_devices())

    plot_by_date(sdk, path, files)
    # compile_plots(files)
    # plot_hist(files)
    # plot_by_age(files)
