import concurrent.futures

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from atriumdb import AtriumSDK, DatasetDefinition
from data_quality import DataValidator
from pat import calclulate_pat
from plotting import plot_pat, plot_pat_hist
from tqdm import tqdm


def process_pat(sdk, dev, itr):
    """
    Processing function for dataset iterator

    :param sdk: AtriumSDK instance
    :param dev: Device ID
    :param itr: Dataset iterator

    :return: Dictionary of pulse arrival times for each patient in device
    """

    results = {}
    total_corrected = 0

    # Progress bar
    pbar = tqdm(total=itr._length, desc="Processing Windows")

    for i, w in enumerate(itr):
        # Check if patient exists in results
        if w.patient_id not in results.keys():
            # Get patient date of birth
            dob = pd.to_datetime(sdk.get_patient_info(w.patient_id)["dob"])

            results[w.patient_id] = {
                "times": np.array([]),
                "pat": np.array([]),
                "naive_pat": np.array([]),
                "dob": dob,
            }

        # Extract specific signals and convert timescale
        ecg, ecg_freq = [(v, k[1]) for k, v in w.signals.items() if "ECG" in k[0]][0]
        ppg, ppg_freq = [(v, k[1]) for k, v in w.signals.items() if "PULS" in k[0]][0]

        ecg["times"] = ecg["times"] / 10**9
        ppg["times"] = ppg["times"] / 10**9

        try:
            pats, naive_pats, corrected = calclulate_pat(ecg, ecg_freq, ppg, ppg_freq)

            total_corrected += corrected

            results[w.patient_id]["times"] = np.concatenate(
                (results[w.patient_id]["times"], pats[:, 0])
            )
            results[w.patient_id]["pat"] = np.concatenate(
                (results[w.patient_id]["pat"], pats[:, 1])
            )
            results[w.patient_id]["naive_pat"] = np.concatenate(
                (results[w.patient_id]["naive_pat"], naive_pats)
            )

        except Exception as e:
            print(f"Error in calculating PAT: {e}")
            continue

        pbar.update(1)

    np.save(f"../data/results/device{dev}_pats.npy", results)
    print(f"Finished processing device {dev}")

    return results


def make_device_itr(
    sdk,
    device,
    window_size_nano=60 * 20 * (10**9),
    gap_tol_nano=0.3 * (10**9),
):
    """
    Creates new SDK instance and iterator for a specific device

    :param sdk: AtriumSDK instance
    :param device: Device ID
    :param window_size_nano: Window size in nanoseconds
    :param gap_tol_nano: Gap tolerance in nanoseconds

    :return: Dictionary of pulse arrival times for each patient in device
    """

    print(f"Building dataset, device: {device}")

    measures = [
        {
            "tag": "MDC_PRESS_BLD_ART_ABP",
            "freq_nhz": 125_000_000_000,
            "units": "MDC_DIM_MMHG",
        },
        {
            "tag": "MDC_ECG_ELEC_POTL_II",
            "freq_nhz": 500_000_000_000,
            "units": "MDC_DIM_MILLI_VOLT",
        },
        {
            "tag": "MDC_PULS_OXIM_PLETH",
            "freq_nhz": 125_000_000_000,
            "units": "MDC_DIM_DIMLESS",
        },
    ]

    definition = DatasetDefinition.build_from_intervals(
        sdk,
        "measures",
        measures=measures,
        device_id_list={device: "all"},
        merge_strategy="intersection",
        gap_tolerance=gap_tol_nano,
    )

    itr = sdk.get_iterator(
        definition,
        window_size_nano,
        window_size_nano,
        num_windows_prefetch=100,
        # cached_windows_per_source=20,
        shuffle=False,
    )

    return itr


if __name__ == "__main__":

    # local_dataset = "/mnt/datasets/atriumdb_abp_estimation_2024_02_05"
    # local_dataset = "/mnt/datasets/ians_data_2024_06_12"

    # Newest dataset with Philips measures (SBP, DBP, MAP) (incomplete, 90%)
    local_dataset = "/mnt/datasets/ian_dataset_2024_07_22"

    sdk = AtriumSDK(dataset_location=local_dataset)
    devices = list(sdk.get_all_devices().keys())
    print(f"Devices: {devices}")

    itr = make_device_itr(sdk, 80)
    process_pat(sdk, 80, itr)

    exit()

    num_cores = 10  # len(devices)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as pp:

        futures = {pp.submit(process, local_dataset, d): d for d in devices}

        for f in concurrent.futures.as_completed(futures):
            print(f.result())

    print("Finished processing")

    measure_tag_list = [
        "MDC_PULS_OXIM_PLETH",  # PPG
        "MDC_ECG_ELEC_POTL_II",  # ECG II
        "MDC_PRESS_BLD_ART",  # ABP I
        "MDC_PRESS_BLD_ART_ABP",  # ABP II
        "MDC_ECG_CARD_BEAT_RATE",  # ECG HR
        "MDC_PLETH_PULS_RATE",  # PPG HR
        "MDC_PULS_RATE",  # PPG HR
        "MDC_PRESS_BLD_ART_ABP_MEAN",
        "MDC_PRESS_BLD_ART_ABP_SYS",
        "MDC_PRESS_BLD_ART_ABP_DIA",
        "MDC_PRESS_BLD_ART_MEAN",
        "MDC_PRESS_BLD_ART_SYS",
        "MDC_PRESS_BLD_ART_DIA",
    ]
