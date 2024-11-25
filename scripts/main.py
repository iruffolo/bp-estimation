import concurrent.futures
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from atriumdb import AtriumSDK, DatasetDefinition
from numpy.polynomial import Polynomial
from scipy.stats import pearsonr, spearmanr
from sklearn import linear_model

from correlation import create_aligned_data
from data_quality import DataValidator
from pat import calclulate_pat
from plotting.slopes import plot_slopes
from plotting.waveforms import plot_waveforms
from sawtooth import fit_sawtooth
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
    log = Logger(dev, num_windows, path="../data/andrew/", verbose=True)

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

        # Filter out by date less than 2023-01-01
        # if ecg["times"][0] > 1640998861:
        #     print(f"Date filter {datetime.fromtimestamp(ecg['times'][0])}")
        #     continue

        try:
            pats, naive_pats, n_corrected, ecg_times, ppg_times = calclulate_pat(
                ecg, ecg_freq, ppg, ppg_freq
            )

            # Window has very sparse measurements, poor quality
            if pats["times"].size < 500:
                log.log_status(WindowStatus.INSUFFICIENT_PATS)
                continue

            # if np.median(pats["values"]) < 0.5 or np.median(pats["values"]) > 2:
            #     print(pats)
            #     print(naive_pats)
            #     print(f"pats: {len(pats)}")
            #     print(f"naive_pats: {len(naive_pats)}")
            #     print(f"matching_peaks: {len(matching_peaks)}")
            #     print(f"ecg_peak_times: {len(ecg_peak_times)}")
            #     print(f"ppg_peak_times: {len(ppg_peak_times)}")
            #     # # Debug plot
            #     plot_waveforms(ecg, ppg, abp, show=True)
            #     exit()

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
            # median = (
            #     pd.DataFrame(synced)
            #     .groupby("bp")
            #     .agg({"pats": ["median", "count"], "naive_pats": ["median", "count"]})
            #     .reset_index()
            # )

            # Filter by minimum number of PAT points per BP value
            # f1 = median[["pats", "bp"]][median["pats"]["count"] > 20]
            # f2 = median[["naive_pats", "bp"]][median["naive_pats"]["count"] > 20]

            # s1 = spearmanr(f1["pats"]["median"], f1["bp"])
            # s2 = spearmanr(f2["naive_pats"]["median"], f2["bp"])

            # Fit lines of best fit
            # l1 = Polynomial.fit(f1["pats"]["median"], f1["bp"], 1, full=True)
            # l2 = Polynomial.fit(f2["naive_pats"]["median"], f2["bp"], 1, full=True)

            # Debug plot
            # plot_slopes(synced, f1, f2, l1[0], l2[0])

            log.log_data(
                w.patient_id,
                sdk.get_patient_info(w.patient_id)["dob"],
                ecg["times"][0],
                n_corrected,
                s1,
                s2,
                hr,
                synced,
            )
            log.log_raw_data(
                {
                    "patient_id": np.full_like(pats["times"], w.patient_id),
                    "ecg_peaks": pats["times"],
                    "pat": pats["values"],
                    "corrected_pat": corrected_pat,
                },
                f"pats",
            )
            log.log_raw_data(
                {
                    "patient_id": np.full_like(naive_pats["times"], w.patient_id),
                    "ecg_peaks": naive_pats["times"],
                    "naive_pat": naive_pats["values"],
                },
                f"naive_pats",
            )
            log.log_raw_data(
                {
                    "patient_id": np.full_like(ecg_times, w.patient_id),
                    "ecg_peaks": ecg_times,
                },
                f"ecg_peaks",
            )
            log.log_raw_data(
                {
                    "patient_id": np.full_like(ppg_times, w.patient_id),
                    "ppg_peaks": ppg_times,
                },
                f"ppg_peaks",
            )
            log.log_raw_data(
                {
                    "patient_id": np.full_like(abp["times"], w.patient_id),
                    "abp_time": abp["times"],
                    "abp_value": abp["values"],
                },
                f"abp",
            )
            log.log_raw_data(
                {
                    "patient_id": np.full_like(sbp["times"], w.patient_id),
                    "sbp_time": sbp["times"],
                    "sbp_value": sbp["values"],
                },
                f"sbp",
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

    # twentytwo = 1640998861 * (10**9)
    # twentythree = 1672531200 * (10**9)

    itr = make_device_itr_all_signals(
        sdk,
        window_size,
        gap_tol,
        device=device,
        pid=None,
        prefetch=50,
        shuffle=False,
        start=None,
        end=None,
    )
    process_pat(sdk, device, itr)

    return True


if __name__ == "__main__":

    # Newest dataset with Philips measures (SBP, DBP, MAP) (incomplete, 90%)
    # local_dataset = "/mnt/datasets/ian_dataset_2024_07_22"

    # Newest dataset
    local_dataset = "/mnt/datasets/ian_dataset_2024_08_26"

    sdk = AtriumSDK(dataset_location=local_dataset)
    print_all_measures(sdk)

    devices = list(sdk.get_all_devices().keys())
    print(f"Devices: {devices}")

    window_size = 5 * 60 * 60  # 30 min
    gap_tol = 30 * 60  # 30 min to reduce overlapping windows with gap tol

    itr = make_device_itr_all_signals(sdk, window_size, gap_tol, 80, shuffle=False)
    process_pat(sdk, 80, itr, early_stop=500)
    exit()

    num_cores = 15  # len(devices)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as pp:

        futures = {
            pp.submit(run, local_dataset, window_size, gap_tol, d): d for d in devices
        }

        for f in concurrent.futures.as_completed(futures):
            print(f.result())

    print("Finished processing")
