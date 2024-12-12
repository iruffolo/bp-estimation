import concurrent.futures
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from atriumdb import AtriumSDK, DatasetDefinition
from pat import peak_detect, rpeak_detect_fast
from utils.atriumdb_helpers import make_device_itr_all_signals, make_device_itr_ecg_ppg
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
    log = Logger(dev, num_windows, path="../data/andrew_peaks/", verbose=True)

    count = 0

    for i, w in enumerate(itr):

        if not w.patient_id:
            log.log_status(WindowStatus.NO_PATIENT_ID)
            continue
        else:
            # Check patient age
            info = sdk.get_patient_info(w.patient_id)
            t = datetime.fromtimestamp(w.start_time / 10**9).year
            dob = datetime.fromtimestamp(info["dob"] / 10**9).year
            age_at_visit = t - dob

            if not (t < datetime(2022, 1, 1).year and age_at_visit == 10):
                print(f"Skipping patient, date: {t}, age: {age_at_visit}")
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
                case signal if "ECG_ELEC" in signal:
                    ecg, ecg_freq = v, freq
                case signal if "PULS_OXIM" in signal:
                    ppg, ppg_freq = v, freq
                case _:
                    pass

        return

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
                f"{w.patient_id}_{date.month}_{date.year}_{age_at_visit}_ecg",
            )
            log.log_raw_data(
                {
                    "ppg_peaks": ppg_peak_times,
                },
                f"{w.patient_id}_{date.month}_{date.year}_{age_at_visit}_ppg",
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

        count += 1

        if early_stop and count >= early_stop:
            break

    # log.save()

    print(f"Finished processing device {dev}")


def run(local_dataset, window_size, gap_tol, device):
    """
    Function to run in parallel
    """
    sdk = AtriumSDK(dataset_location=local_dataset)

    itr = make_device_itr_ecg_ppg(
        sdk,
        window_size,
        gap_tol,
        device=device,
        prefetch=10,
    )
    save_peaks(sdk, device, itr, early_stop=10)

    return True


if __name__ == "__main__":

    # Newest dataset with Philips measures (SBP, DBP, MAP) (incomplete, 90%)
    # local_dataset = "/mnt/datasets/ian_dataset_2024_07_22"

    # Newest dataset
    local_dataset = "/mnt/datasets/ian_dataset_2024_08_26"

    sdk = AtriumSDK(dataset_location=local_dataset)

    devices = list(sdk.get_all_devices().keys())
    print(f"Devices: {devices}")

    window_size = 1 * 60 * 60  # 30 min
    gap_tol = 30 * 60  # 30 min to reduce overlapping windows with gap tol

    # itr = make_device_itr_ecg_ppg(sdk, window_size, gap_tol, device=80)
    # save_peaks(sdk, 80, itr, early_stop=50)
    # exit()

    num_cores = 15  # len(devices)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as pp:

        futures = {
            pp.submit(run, local_dataset, window_size, gap_tol, d): d for d in devices
        }

        for f in concurrent.futures.as_completed(futures):
            print(f.result())

    print("Finished processing")
