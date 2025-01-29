import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from atriumdb import AtriumSDK
from sawtooth import fit_sawtooth
from scipy.stats import pearsonr, spearmanr
from utils.atriumdb_helpers import get_all_patient_data, print_all_measures


def _convert_to_np(data):
    """
    Convert every element in dict to np array
    """

    for key in data:
        data[key] = np.array(data[key])

    return data


def create_aligned_data(pats, n_pats, times, bp, bp_lag=6, max_offset=10):
    """
    Align PAT data with BP data
    """

    # print(f"Creating aligned data with offset: {bp_lag}s")
    synced = {"times": [], "pats": [], "naive_pats": [], "bp": []}

    for i, (pat, n_pat, t) in enumerate(zip(pats, n_pats, times)):
        try:
            idx = np.where(
                (bp["times"] - bp_lag >= t) & (bp["times"] - t < max_offset)
            )[0][0]

            # Remove nans
            if np.isnan(pat) or np.isnan(bp["values"][idx]):
                continue

            synced["times"].append(t)
            synced["pats"].append(pat)
            synced["naive_pats"].append(n_pat)
            synced["bp"].append(bp["values"][idx])

            # print(f"Pat: {pat}, time: {t}")
            # print(f"BP times: {bp['times'][idx]}")
            # print(f"Added pat {i} with pat: {pat}, bp: {bp['values'][idx]}")
            # print(f"Time difference: {bp['times'][idx] + offset - t}")

        except IndexError:
            print(f"Failed to align pat, missing BP data for pat {i}")
            return _convert_to_np(synced)
        except Exception as e:
            print(f"Failed to align pat: {e}")
            return _convert_to_np(synced)

    return _convert_to_np(synced)


def calc_correlation(synced):
    """
    Calculate correlation between PAT and BP
    """

    pats = synced[:, 1]
    n_pats = synced[:, 2]
    bp = synced[:, 3]

    return (
        pearsonr(pats, bp),
        pearsonr(n_pats, bp),
        spearmanr(pats, bp),
        spearmanr(n_pats, bp),
    )


if __name__ == "__main__":

    print("Calculating ABP correlation")

    local_dataset = "/mnt/datasets/ian_dataset_2024_07_22"
    sdk = AtriumSDK(dataset_location=local_dataset)

    print_all_measures(sdk)

    num_patients = 0
    num_pats = 0
    total_corrected = 0
    failed_windows = 0

    # bins = np.linspace(0, 4, 5000)
    # all_pats, bins = np.histogram([], bins=bins)
    # all_pats_naive = np.histogram([], bins=bins)[0]

    files = os.listdir("../data/results")
    for f in files[0:1]:

        data = np.load(f"../data/results/{f}", allow_pickle=True).item()

        for p in list(data.keys())[6:10]:
            print(f"Patient: {p}")

            sbp = get_all_patient_data(sdk, p)
            synced = create_aligned_data(
                data[p]["pat"], data[p]["naive_pat"], data[p]["times"], sbp
            )
            if not synced.size > 0:
                continue

            fix, ax = plt.subplots(4, figsize=(10, 10), sharex=True)

            # Get Sawtooth and plot
            st, params = fit_sawtooth(synced[:, 0], synced[:, 1], True)
            corrected_pat = synced[:, 1] - st + params[2]

            ax[0].plot(synced[:, 0], synced[:, 3], ".")
            ax[0].set_title("SBP")
            ax[0].set_xlabel("Time (s)")

            ax[1].plot(synced[:, 0], synced[:, 2], ".")
            ax[1].set_title("Naive PAT")
            ax[1].set_xlabel("Time (s)")

            ax[2].plot(synced[:, 0], synced[:, 1], ".")
            ax[2].set_title("PAT")
            ax[2].set_xlabel("Time (s)")

            ax[3].plot(synced[:, 0], corrected_pat, ".")
            ax[3].set_title("Corrected PAT (Sawtooth)")
            ax[3].set_xlabel("Time (s)")

            plt.show()

            pearson, spearman, n_pearson, n_spearman = calc_correlation(synced)
            print(f"Naive Pearson: {n_pearson}, Naive Spearman: {n_spearman}")
            print(f"Pearson: {pearson}, Spearman: {spearman}")
            print(f"Sawtooth Pearson {pearsonr(corrected_pat, synced[:, 3])}")
            print(f"Sawtooth Spearman {spearmanr(corrected_pat, synced[:, 3])}")

            num_patients += 1
            num_pats += data[p]["pat"].shape[0]
            total_corrected += data[p]["total_corrected"]
            failed_windows += data[p]["failed_windows"]

            # all_pats += np.histogram(data[p]["pat"], bins=bins)[0]
            # all_pats_naive += np.histogram(data[p]["naive_pat"], bins=bins)[0]

    print(f"Number of patients: {num_patients}")
    print(f"Number of pats: {num_pats}")
    print(f"Total corrected: {total_corrected}")
    print(f"Failed windows: {failed_windows}")

    # fig, ax = plt.subplots(1, figsize=(10, 10))
    # ax.stairs(all_pats, bins, fill=True)
    #
    # plt.tight_layout()
    # plt.show()
