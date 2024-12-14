import concurrent.futures
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from atriumdb import AtriumSDK, DatasetDefinition

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

    num_windows = early_stop if early_stop else itr._length
    log = Logger(dev, num_windows, path="../data/beat_matching/", verbose=True)

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
            # if not (t < datetime(2022, 1, 1).year and age_at_visit == 10):
        #         print(f"Skipping patient dev {dev}, date: {t}, age: {age_at_visit}")
        #         continue
        print(f"Processing patient dev {dev}, date: {t}, age: {age_at_visit}")

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
        if v["expected_count"] * 0.2 > v["actual_count"]:
            log.log_status(WindowStatus.INCOMPLETE_WINDOW)
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ecg_peak_times = rpeak_detect_fast(
                    ecg["times"], ecg["values"], ecg_freq
                )
                ppg_peak_times = peak_detect(ppg["times"], ppg["values"], ppg_freq)

            date = datetime.fromtimestamp(ecg["times"][0])

            ssize = 6
            matching_beats = beat_matching(ecg_peak_times, ppg_peak_times, ssize=ssize)
            print(f"Matched beats: {len(matching_beats)}")

            # Create a df for all possible PAT values
            all_pats = pd.DataFrame(
                [m.possible_pats for m in matching_beats],
                columns=[f"{i + 1} beats" for i in range(ssize)],
            )

            all_pats["bm_pat"] = [m.possible_pats[m.n_peaks] for m in matching_beats]
            all_pats["confidence"] = [m.confidence for m in matching_beats]
            all_pats["times"] = [ecg_peak_times[m.ecg_peak] for m in matching_beats]
            all_pats["beats_skipped"] = [m.n_peaks for m in matching_beats]

            correct_pats(all_pats, matching_beats, pat_range=0.300)

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
            log.log_raw_data(
                all_pats, f"{w.patient_id}_{date.month}_{date.year}_{age_at_visit}_pats"
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
            print(f"Unexpected failure: {e}")
            log.log_status(WindowStatus.UNEXPECTED_FAILURE)

        if early_stop and i >= early_stop:
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
        shuffle=True,
    )
    save_pats(sdk, device, itr)

    return True


if __name__ == "__main__":

    # Newest dataset with Philips measures (SBP, DBP, MAP) (incomplete, 90%)
    # local_dataset = "/mnt/datasets/ian_dataset_2024_07_22"

    # Newest dataset
    local_dataset = "/mnt/datasets/ian_dataset_2024_08_26"

    sdk = AtriumSDK(dataset_location=local_dataset)

    devices = list(sdk.get_all_devices().keys())
    print(f"Devices: {devices}")

    window_size = 5 * 60 * 60  # 60 min
    gap_tol = 60 * 60  # 30 min to reduce overlapping windows with gap tol

    # itr = make_device_itr_ecg_ppg(sdk, window_size, gap_tol, device=80)
    # save_pats(sdk, 80, itr, early_stop=50)
    # exit()

    num_cores = 15  # len(devices)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as pp:

        futures = {
            pp.submit(run, local_dataset, window_size, gap_tol, d): d for d in devices
        }

        for f in concurrent.futures.as_completed(futures):
            print(f.result())

    print("Finished processing")
