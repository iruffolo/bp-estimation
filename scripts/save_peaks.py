import concurrent.futures
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from atriumdb import AtriumSDK, DatasetDefinition
from pat import peak_detect, rpeak_detect_fast
from utils.atriumdb_helpers import make_device_itr_all_signals, print_all_measures
from utils.logger import Logger, WindowStatus


def save_peaks(sdk, dev, itr, early_stop=None):
    """
    Processing function for dataset iterator

    :param sdk: AtriumSDK instance
    :param dev: Device ID
    :param itr: Dataset iterator

    :return: Dictionary of pulse arrival times for each patient in device
    """

    num_windows = early_stop if early_stop else itr._length
    log = Logger(dev, num_windows, path="../data/peaks/", verbose=True)

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
                case signal if "SYS" in signal:
                    sbp = v
                case signal if "ECG_ELEC" in signal:
                    ecg, ecg_freq = v, freq
                case signal if "PULS_OXIM" in signal:
                    ppg, ppg_freq = v, freq
                case _:
                    pass

        # Ensure window data is valid (gap tol can create bad windows)
        # if v["expected_count"] * 0.5 > v["actual_count"]:
        #     log.log_status(WindowStatus.INCOMPLETE_WINDOW)
        #     continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ecg_peak_times = rpeak_detect_fast(
                    ecg["times"], ecg["values"], ecg_freq
                )
                ppg_peak_times = peak_detect(ppg["times"], ppg["values"], ppg_freq)

            date = datetime.fromtimestamp(ecg["times"][0])

            log.log_raw_data(
                {
                    "ecg_peaks": ecg_peak_times,
                },
                f"{w.patient_id}_{date.month}_{date.year}_ecg_peaks",
            )
            log.log_raw_data(
                {
                    "ppg_peaks": ppg_peak_times,
                },
                f"{w.patient_id}_{date.month}_{date.year}_ppg_peaks",
            )
            log.log_raw_data(
                {
                    "sbp_time": sbp["times"],
                    "sbp_value": sbp["values"],
                },
                f"{w.patient_id}_{date.month}_{date.year}_sbp",
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

    itr = make_device_itr_all_signals(
        sdk,
        window_size,
        gap_tol,
        device=device,
        pid=None,
        prefetch=10,
        shuffle=False,
        start=None,
        end=None,
    )
    save_peaks(sdk, device, itr)

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

    window_size = 2 * 60 * 60  # 30 min
    gap_tol = 30 * 60  # 30 min to reduce overlapping windows with gap tol

    # itr = make_device_itr_all_signals(sdk, window_size, gap_tol, 80, shuffle=False)
    # save_peaks(sdk, 80, itr, early_stop=50)
    # exit()

    num_cores = 10  # len(devices)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as pp:

        futures = {
            pp.submit(run, local_dataset, window_size, gap_tol, d): d for d in devices
        }

        for f in concurrent.futures.as_completed(futures):
            print(f.result())

    print("Finished processing")
