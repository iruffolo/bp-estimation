import concurrent.futures
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from atriumdb import AtriumSDK, DatasetDefinition
from beat_matching import beat_matching, correct_pats, peak_detect, rpeak_detect_fast
from kf_sawtooth import calc_sawtooth
from utils.atriumdb_helpers import (
    make_device_itr_all_signals,
    make_device_itr_ecg_ppg,
    print_all_measures,
)
from utils.logger import Logger, WindowStatus


def save_pats(sdk, dev, itr, early_stop=None):
    """
    Processing function for dataset iterator

    :param sdk: AtriumSDK instance
    :param dev: Device ID
    :param itr: Dataset iterator

    :return: Dictionary of pulse arrival times for each patient in device
    """

    num_windows = early_stop if early_stop else len(itr)
    log = Logger(
        dev,
        num_windows,
        path="/home/iruffolo/dev/bp-estimation/data/result_histograms/",
        verbose=True,
    )

    for i, w in enumerate(itr):

        if not w.patient_id:
            log.log_status(WindowStatus.NO_PATIENT_ID)
            continue
        else:
            # Check patient age
            info = sdk.get_patient_info(w.patient_id)
            t = datetime.fromtimestamp(w.start_time / 10**9)
            dob = datetime.fromtimestamp(info["dob"] / 10**9)
            age_at_visit = (t - dob).days

        print(
            f"Processing patient {w.patient_id} dev {dev}, date: {t}, age: {age_at_visit} days"
        )

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

        # Ensure window data is valid (gap tol can create bad windows)
        if v["expected_count"] * 0.05 > v["actual_count"]:
            print(f"Incomplete window {v['actual_count']}")
            log.log_status(WindowStatus.INCOMPLETE_WINDOW)
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ecg_peak_times = rpeak_detect_fast(
                    ecg["times"], ecg["values"], ecg_freq
                )
                ppg_peak_times = peak_detect(ppg["times"], ppg["values"], ppg_freq)

        # Peak detection faliled to detect enough peaks in calculate_pat
        except AssertionError as e:
            print(f"Signal quality issue: {e}")
            if "ECG" in str(e):
                log.log_status(WindowStatus.POOR_ECG_QUALITY)
            elif "PPG" in str(e):
                log.log_status(WindowStatus.POOR_PPG_QUALITY)
            elif "BM" in str(e):
                log.log_status(WindowStatus.BM_FAILED)
            else:
                print(f"Unexpected assert failure: {e}")
                log.log_status(WindowStatus.UNEXPECTED_FAILURE)

        except Exception as e:
            print(f"Unexpected failure: {e}")
            log.log_status(WindowStatus.UNEXPECTED_FAILURE)

        if early_stop and i >= early_stop:
            break

    log.log_pickle(hists, "histograms")
    log.save_log()

    print(f"Finished processing device {dev}")


def run(
    local_dataset,
    window_size,
    gap_tol,
    device,
    callback,
    log_path,
    start_nano=None,
    end_nano=None,
    early_stop=None,
):
    """
    Function to run in parallel. Creates an iterator for a device, and passes
    data back to callback
    """
    sdk = AtriumSDK(dataset_location=local_dataset)

    num_windows = early_stop if early_stop else len(itr)
    log = Logger(
        device,
        num_windows,
        path=log_path,
        verbose=True,
    )

    itr = make_device_itr_ecg_ppg(
        sdk,
        window_size,
        gap_tol,
        device=device,
        prefetch=10,
        cache=10,
        shuffle=False,
        start_nano=start_nano,
        end_nano=end_nano,
    )

    for i, w in enumerate(itr):

        if not w.patient_id:
            log.log_status(WindowStatus.NO_PATIENT_ID)
            continue
        else:
            # Check patient age
            dob = datetime.fromtimestamp(
                sdk.get_patient_info(w.patient_id)["dob"] / 10**9
            )
            t = datetime.fromtimestamp(w.start_time / 10**9)
            age = (t - dob).days

        print(f"Processing pid: {w.patient_id} dev: {dev}, date: {t}, age: {age}d")

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

        # Ensure window data is valid (gap tol can create bad windows)
        if v["expected_count"] * 0.05 > v["actual_count"]:
            print(f"Incomplete window {v['actual_count']}")
            log.log_status(WindowStatus.INCOMPLETE_WINDOW)
            continue

        callback(ecg, ecg_freq, ppg, ppg_freq, log)

    return True


if __name__ == "__main__":

    # Newest dataset
    # local_dataset = "/home/ian/dev/datasets/ian_dataset_2024_08_26"
    local_dataset = "/mnt/datasets/ian_dataset_2024_08_26"
    sdk = AtriumSDK(dataset_location=local_dataset)

    log_path = "/home/iruffolo/dev/bp-estimation/data/result_histograms/"

    devices = list(sdk.get_all_devices().keys())
    print(f"Devices: {devices}")

    window_size = 1 * 60 * 60
    gap_tol = 5 * 60

    start = None
    end = datetime(year=2022, month=5, day=1).timestamp() * (10**9)

    early_stop = None
    # start = datetime(year=2022, month=8, day=1).timestamp() * (10**9)
    # end = None
    # run(local_dataset, window_size, gap_tol, 80, start, end)
    # exit()

    num_cores = 15

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as pp:

        futures = {
            pp.submit(
                run, local_dataset, window_size, gap_tol, d, start, end, early_stop
            ): d
            for d in devices
        }

        for f in concurrent.futures.as_completed(futures):
            print(f.result())

    print("Finished processing")
