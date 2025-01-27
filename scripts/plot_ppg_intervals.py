import concurrent.futures
import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from atriumdb import AtriumSDK, DatasetDefinition
from kf_sawtooth import calc_sawtooth
from utils.atriumdb_helpers import make_device_itr_all_signals, make_device_itr_ecg_ppg
from utils.logger import Logger, WindowStatus
from visualization.beat_matching import (
    beat_matching,
    correct_pats,
    peak_detect,
    rpeak_detect_fast,
)


def save_pats(sdk, dev, itr, early_stop=None):
    """
    Processing function for dataset iterator

    :param sdk: AtriumSDK instance
    :param dev: Device ID
    :param itr: Dataset iterator

    :return: Dictionary of pulse arrival times for each patient in device
    """

    num_windows = early_stop if early_stop else len(itr)
    log = Logger(dev, num_windows, path="../data/st_correction_post/", verbose=True)

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

            fig, ax = plt.subplots(1, 3)

            csum = np.cumsum(np.diff(ppg_peak_times))
            print(csum)

            x = ppg_peak_times[:-1] - ppg_peak_times[0]

            poly = np.polyfit(x, csum, 1)
            poly1d = np.poly1d(poly)

            y = csum - poly1d(x)

            ax[0].plot(ppg_peak_times[:-1], np.diff(ppg_peak_times))
            ax[1].scatter(x, csum, s=0.5, marker="x")
            ax[1].plot(x, poly1d(x))
            ax[1].set_title("cumsum")
            ax[2].scatter(x, y, s=0.5)
            ax[2].set_title("detrended cumsum")

            plt.show()

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

    log.save_log()

    print(f"Finished processing device {dev}")


def run(local_dataset, window_size, gap_tol, device, start_nano=None, end_nano=None):
    """
    Function to run in parallel
    """
    sdk = AtriumSDK(dataset_location=local_dataset)

    itr = make_device_itr_ecg_ppg(
        sdk,
        window_size,
        gap_tol,
        device=device,
        prefetch=1,
        shuffle=False,
        start_nano=start_nano,
        end_nano=end_nano,
    )
    save_pats(sdk, device, itr)

    return True


if __name__ == "__main__":

    # Newest dataset
    # local_dataset = "/mnt/datasets/ian_dataset_2024_08_26"
    local_dataset = "/home/ian/dev/datasets/ian_dataset_2024_08_26"

    sdk = AtriumSDK(dataset_location=local_dataset)

    devices = list(sdk.get_all_devices().keys())
    print(f"Devices: {devices}")

    window_size = 1 * 10 * 60
    gap_tol = 30

    # start = datetime(year=2022, month=8, day=1).timestamp() * (10**9)
    # end = None

    start = None
    end = datetime(year=2022, month=6, day=1).timestamp() * (10**9)

    run(local_dataset, window_size, gap_tol, 80, start, end)
