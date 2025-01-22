import concurrent.futures
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from atriumdb import AtriumSDK, DatasetDefinition
from kf_sawtooth import calc_sawtooth
from utils.atriumdb_helpers import (
    make_device_itr_all_signals,
    make_device_itr_ecg_ppg,
    print_all_measures,
)
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
    log = Logger(
        dev,
        num_windows,
        path="/home/iruffolo/dev/bp-estimation/data/result_histograms/",
        verbose=True,
    )

    age_bins = np.linspace(0, 7000, 7001)

    bins = 5000
    bin_range = (0, 5)

    _, edges = np.histogram([], bins=bins, range=bin_range)
    hists = {
        "naive": {
            age: np.histogram([], bins=bins, range=bin_range)[0] for age in age_bins
        },
        "bm": {
            age: np.histogram([], bins=bins, range=bin_range)[0] for age in age_bins
        },
        "bm_st1": {
            age: np.histogram([], bins=bins, range=bin_range)[0] for age in age_bins
        },
        "bm_st1_st2": {
            age: np.histogram([], bins=bins, range=bin_range)[0] for age in age_bins
        },
    }

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

            date = datetime.fromtimestamp(ecg["times"][0])

            assert ecg_peak_times.size > 500, "Not enough ECG peaks found"
            assert ppg_peak_times.size > 500, "Not enough PPG peaks found"

            ssize = 6
            matching_beats = beat_matching(ecg_peak_times, ppg_peak_times, ssize=ssize)
            assert len(matching_beats) > 0, "BM failed to find any matching beats"

            log.log_status(WindowStatus.TOTAL_BEATS, len(ecg_peak_times))
            log.log_status(
                WindowStatus.TOTAL_BEATS_DROPPED,
                len(ecg_peak_times) - len(matching_beats),
            )

            # Create a df for all possible PAT values
            all_pats = pd.DataFrame(
                [m.possible_pats for m in matching_beats],
                columns=[f"{i + 1} beats" for i in range(ssize)],
            )
            all_pats["bm_pat"] = [m.possible_pats[m.n_peaks] for m in matching_beats]
            all_pats["confidence"] = [m.confidence for m in matching_beats]
            all_pats["times"] = [ecg_peak_times[m.ecg_peak] for m in matching_beats]
            all_pats["beats_skipped"] = [m.n_peaks for m in matching_beats]
            all_pats["age_days"] = all_pats["times"].apply(
                lambda x: (datetime.fromtimestamp(x) - dob).days
            )

            correct_pats(all_pats, matching_beats, pat_range=0.100)
            df = all_pats[all_pats["valid_correction"] > 0]

            fn = f"{w.patient_id}_{dev}_{i}"
            cdata, p1, p2 = calc_sawtooth(df["times"], df["corrected_bm_pat"], fn)

            st1 = pd.DataFrame(cdata["st1"])
            st2 = pd.DataFrame(cdata["st2"])

            st1["age_days"] = st1["times"].apply(
                lambda x: (datetime.fromtimestamp(x) - dob).days
            )
            st2["age_days"] = st2["times"].apply(
                lambda x: (datetime.fromtimestamp(x) - dob).days
            )

            for day in all_pats["age_days"][all_pats["age_days"] <= 7000].unique():
                naive = np.histogram(
                    all_pats["1 beats"][all_pats["age_days"] == day],
                    bins=bins,
                    range=bin_range,
                )[0]
                hists["naive"][day] += naive

                bm = np.histogram(
                    df["corrected_bm_pat"][df["age_days"] == day],
                    bins=bins,
                    range=bin_range,
                )[0]
                hists["bm"][day] += bm

                bm_st1 = np.histogram(
                    st1["values"][st1["age_days"] == day],
                    bins=bins,
                    range=bin_range,
                )[0]
                hists["bm_st1"][day] += bm_st1

                bm_st1_st2 = np.histogram(
                    st2["values"][st2["age_days"] == day],
                    bins=bins,
                    range=bin_range,
                )[0]
                hists["bm_st1_st2"][day] += bm_st1_st2

            log.log_raw_data(p1, f"st1_params")
            log.log_raw_data(p2, f"st2_params")

            log.log_status(WindowStatus.SUCCESS)

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
    start_nano=None,
    end_nano=None,
    early_stop=None,
):
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
        cache=10,
        shuffle=False,
        start_nano=start_nano,
        end_nano=end_nano,
    )
    save_pats(sdk, device, itr, early_stop=early_stop)

    return True


if __name__ == "__main__":

    # Newest dataset
    # local_dataset = "/home/ian/dev/datasets/ian_dataset_2024_08_26"
    local_dataset = "/mnt/datasets/ian_dataset_2024_08_26"

    sdk = AtriumSDK(dataset_location=local_dataset)
    # print_all_measures(sdk)

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
