import concurrent.futures
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from atriumdb import AtriumSDK, DatasetDefinition
from save_pats import SavePats
from utils.atriumdb_helpers import (
    make_device_itr_all_signals,
    make_device_itr_ecg_ppg,
    print_all_measures,
)
from utils.logger import Logger, WindowStatus


def run_device(
    local_dataset,
    window_size,
    gap_tol,
    device,
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

    itr = make_device_itr_ecg_ppg(
        sdk,
        window_size,
        gap_tol,
        device=device,
        shuffle=True,
        start_nano=start_nano,
        end_nano=end_nano,
    )

    num_windows = early_stop if early_stop else len(itr)
    log = Logger(
        device,
        num_windows,
        path=log_path,
        verbose=True,
    )

    sp = SavePats(device, log)

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

        print(f"Processing pid: {w.patient_id} dev: {device}, date: {t}, age: {age}d")

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
            sp.process_window(ecg, ecg_freq, ppg, ppg_freq, dob, w.patient_id)
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
                raise

        # except Exception as e:
        #     print(f"Unexpected failure: {e}")
        #     log.log_status(WindowStatus.UNEXPECTED_FAILURE)
        #     raise

    return True


if __name__ == "__main__":

    # Newest dataset
    # local_dataset = "/home/ian/dev/datasets/ian_dataset_2024_08_26"
    local_dataset = "/mnt/datasets/ian_dataset_2024_08_26"
    sdk = AtriumSDK(dataset_location=local_dataset)

    log_path = "/home/iruffolo/dev/bp-estimation/data/result_histograms/"

    devices = list(sdk.get_all_devices().keys())
    print(f"Devices: {devices}")

    # Params for run
    num_cores = len(devices)
    window_size = 1 * 60 * 60
    gap_tol = 5 * 60
    start = None
    end = datetime(year=2022, month=5, day=1).timestamp() * (10**9)
    early_stop = None
    # start = datetime(year=2022, month=8, day=1).timestamp() * (10**9)
    # end = None

    # run_device(local_dataset, window_size, gap_tol, 80, log_path, start, end)
    # exit()

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as pp:

        futures = {
            pp.submit(
                run_device,
                local_dataset,
                window_size,
                gap_tol,
                d,
                log_path,
                start,
                end,
                early_stop,
            ): d
            for d in devices
        }

        for f in concurrent.futures.as_completed(futures):
            print(f.result())

    print("Finished processing")
