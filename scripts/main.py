import concurrent.futures
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from atriumdb import AtriumSDK, DatasetDefinition
from correlation import create_aligned_data
from data_quality import DataValidator
from numpy.polynomial import Polynomial
from pat import calclulate_pat
from plotting.slopes import plot_slopes
from plotting.waveforms import plot_waveforms
from sawtooth import fit_sawtooth
from scipy.stats import pearsonr, spearmanr
from sklearn import linear_model
from utils.atriumdb_helpers import (
    make_device_itr,
    make_device_itr_all_signals,
    print_all_measures,
)
from utils.logger import Logger, WindowStatus


def process_pat(sdk, dev, itr, early_stop=None):
    """
    Processing function for dataset iterator

    :param sdk: AtriumSDK instance
    :param dev: Device ID
    :param itr: Dataset iterator

    :return: Dictionary of pulse arrival times for each patient in device
    """

    num_windows = early_stop if early_stop else itr._length
    log = Logger(dev, num_windows, path="../data/results/", verbose=True)

    for i, w in enumerate(itr):

        if not w.patient_id:
            log.log_status(WindowStatus.NO_PATIENT_ID)
            continue

        # Extract data from window and validate
        for (signal, freq, _), v in w.signals.items():
            # Skip signals without data (ABP, SYS variants)
            if v["actual_count"] == 0:
                continue

            # Convert to s
            v["times"] = v["times"] / (10**9)

            # Extract specific signals
            match signal:
                case signal if "BEAT_RATE" in signal:
                    hr = v
                case signal if "SYS" in signal:
                    sbp = v
                case signal if "ECG_ELEC" in signal:
                    ecg, ecg_freq = v, freq
                case signal if "PULS_OXIM" in signal:
                    ppg, ppg_freq = v, freq
                case "MDC_PRESS_BLD_ART" | "MDC_PRESS_BLD_ART_ABP":
                    abp, abp_freq = v, freq
                case _:
                    pass

        # Ensure window data is valid (gap tol can create bad windows)
        if v["expected_count"] * 0.5 > v["actual_count"]:
            log.log_status(WindowStatus.INCOMPLETE_WINDOW)
            continue

        try:
            pats, naive_pats, n_corrected = calclulate_pat(ecg, ecg_freq, ppg, ppg_freq)

            # Window has very sparse measurements, poor quality
            if pats["times"].size < 500:
                log.log_status(WindowStatus.INSUFFICIENT_PATS)
                continue

            # Get Sawtooth and correct PATs
            st, params = fit_sawtooth(pats["times"], pats["values"], plot=False)
            corrected_pat = pats["values"] - st + params[2]

            # Align pats with SBP values
            synced = create_aligned_data(
                corrected_pat, naive_pats["values"], pats["times"], sbp
            )

            if synced["times"].size < 300:
                log.log_status(WindowStatus.FAILED_BP_ALIGNMENT)
                continue

            # Get Correlation with SBP
            s1 = spearmanr(synced["pats"], synced["bp"])
            s2 = spearmanr(synced["naive_pats"], synced["bp"])

            # Calculate medians for better line of best fit
            median = (
                pd.DataFrame(synced)
                .groupby("bp")
                .agg({"pats": ["median", "count"], "naive_pats": ["median", "count"]})
                .reset_index()
            )

            # Filter by minimum number of PAT points per BP value
            f1 = median[["pats", "bp"]][median["pats"]["count"] > 20]
            f2 = median[["naive_pats", "bp"]][median["naive_pats"]["count"] > 20]

            # Fit lines of best fit
            l1 = Polynomial.fit(f1["pats"]["median"], f1["bp"], 1, full=True)
            l2 = Polynomial.fit(f2["naive_pats"]["median"], f1["bp"], 1, full=True)

            # Debug plot
            # plot_slopes(synced, f1, f2, l1[0], l2[0])

            log.log_data(
                w.patient_id,
                sdk.get_patient_info(w.patient_id)["dob"],
                ecg["times"][0],
                n_corrected,
                s1,
                s2,
                l1,
                l2,
                hr,
                synced,
            )
            log.log_status(WindowStatus.SUCCESS)

        # Peak detection faliled to detect enough peaks in calculate_pat
        except AssertionError as e:
            print(f"Signal quality issue: {e}")
            if "ECG" in str(e):
                log.log_status(WindowStatus.POOR_ECG_QUALITY)
            if "PPG" in str(e):
                log.log_status(WindowStatus.POOR_PPG_QUALITY)
            else:
                log.log_status(WindowStatus.UNEXPECTED_FAILURE)

            # # Debug plot
            # plot_waveforms(ecg, ppg, abp, show=True)

        except Exception as e:
            print(f"Unexpected failure {e}")
            log.log_status(WindowStatus.UNEXPECTED_FAILURE)

        if early_stop and i >= early_stop:
            break

    log.save()

    print(f"Finished processing device {dev}")


def run(local_dataset, window_size, gap_tol, device):
    """
    Function to run in parallel
    """
    sdk = AtriumSDK(dataset_location=local_dataset)
    itr = make_device_itr_all_signals(sdk, device, window_size, gap_tol)
    process_pat(sdk, device, itr)

    return True


if __name__ == "__main__":

    # Newest dataset with Philips measures (SBP, DBP, MAP) (incomplete, 90%)
    # local_dataset = "/mnt/datasets/ian_dataset_2024_07_22"

    # Newest dataset
    local_dataset = "/mnt/datasets/ian_dataset_2024_08_15"

    sdk = AtriumSDK(dataset_location=local_dataset)
    print_all_measures(sdk)

    devices = list(sdk.get_all_devices().keys())
    print(f"Devices: {devices}")

    window_size = 60 * 60  # 60 min
    gap_tol = 5 * 60  # 5 min to reduce overlapping windows with gap tol

    itr = make_device_itr_all_signals(sdk, 80, window_size, gap_tol, 1)
    process_pat(sdk, 80, itr, early_stop=2)
    exit()

    num_cores = 10  # len(devices)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as pp:

        futures = {
            pp.submit(run, local_dataset, window_size, gap_tol, d): d for d in devices
        }

        for f in concurrent.futures.as_completed(futures):
            print(f.result())

    print("Finished processing")
