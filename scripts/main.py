import concurrent.futures

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from atriumdb import AtriumSDK, DatasetDefinition
from data_quality import DataValidator
from pat import calclulate_pat
from plotting import plot_pat, plot_pat_hist


def process_pat(itr, dev, sdk):
    """
    Processing function for dataset iterator

    :param itr: Dataset iterator
    :param dev: Device ID

    :return: Dictionary of pulse arrival times for each patient in device
    """

    p = {}
    # dv = DataValidator()
    bins = np.linspace(0, 4, 5000)

    for i, w in enumerate(itr):
        print(f"Processing window {i}... Patient {w.patient_id}")

        # Extract specific signals
        ecg_data, ecg_freq = [
            (v, k[1] / 10**9) for k, v in w.signals.items() if "ECG" in k[0]
        ][0]
        ppg_data, ppg_freq = [
            (v, k[1] / 10**9) for k, v in w.signals.items() if "PULS" in k[0]
        ][0]

        print(w)

        # if (np.isnan(ecg_data["values"]).any()) or (np.isnan(ppg_data["values"]).any()):
        #     print("Skipping window due to NaNs")
        #     continue

        dob = pd.to_datetime(sdk.get_patient_info(w.patient_id)["dob"])

        ecg_data["times"] = ecg_data["times"] / 10**9
        ppg_data["times"] = ppg_data["times"] / 10**9

        try:
            pats, ecg_peak_times, ppg_peak_times, n_cleaned = calclulate_pat(
                ecg_data, ecg_freq, ppg_data, ppg_freq
            )

            ratio = pats.shape[0] / ecg_peak_times.shape[0]

            # Another quality check
            if ratio > 0.5:

                if w.patient_id not in p.keys():
                    p[w.patient_id] = {
                        "pat": np.histogram([], bins=bins)[0],
                        "dob": dob,
                        "visit_time": ecg_data["times"][0],
                    }

                # p[w.patient_id]["pat"].append(pats[:, 1])
                p[w.patient_id]["pat"] += np.histogram(pats[:, 1], bins=bins)[0]

                # if i < 10:
                #     plot_pat(
                #         ecg_data,
                #         ecg_peak_times,
                #         ppg_data,
                #         ppg_peak_times,
                #         pats,
                #         show=False,
                #         save=True,
                #         patient_id=w.patient_id,
                #         device_id=dev,
                #     )

        except Exception as e:
            print(f"Error in calculating PAT: {e}")
            continue

        if i > 5000:
            break

    # Flatten all the patient data
    # for k, v in p.items():
    #     p[k]["pat"] = np.concatenate(v["pat"])

    np.save(f"raw_data/datesplit/results_{dev}.npy", p)
    print(f"Finished processing device {dev}")

    return True


def process(
    dataset_location,
    device,
    window_size_nano=60 * 10 * (10**9),
    gap_tol_nano=0.5 * (10**9),
):
    """
    Creates new SDK instance and iterator for device

    :param sdk: AtriumSDK instance
    :param device: Device ID
    :param window_size_nano: Window size in nanoseconds
    :param gap_tol_nano: Gap tolerance in nanoseconds

    :return: Dictionary of pulse arrival times for each patient in device
    """

    sdk = AtriumSDK(dataset_location=dataset_location)

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

    print(f"Building dataset, device: {device}")

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
        cached_windows_per_source=10,
        shuffle=True,
    )

    return process_pat(itr, device, sdk)


if __name__ == "__main__":

    local_dataset = "/mnt/datasets/atriumdb_abp_estimation_2024_02_05"

    sdk = AtriumSDK(dataset_location=local_dataset)
    devices = list(sdk.get_all_devices().keys())
    print(f"Devices: {devices}")

    num_cores = 10  # len(devices)

    # process(local_dataset, 74)
    # exit()

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as pp:

        futures = {pp.submit(process, local_dataset, d): d for d in devices}

        for f in concurrent.futures.as_completed(futures):
            print(f.result())

    print("Finished processing")
