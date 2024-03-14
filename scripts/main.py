import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import concurrent.futures

from atriumdb import AtriumSDK, DatasetDefinition
from data_quality import DataValidator
from pat import align_peaks, plot_pat, plot_pat_hist


def process_pat(itr, dev):
    """
    Processing function for dataset iterator

    :param itr: Dataset iterator
    :param dev: Device ID

    :return: Dictionary of pulse arrival times for each patient in device
    """

    p = {}
    # dv = DataValidator()

    for i, w in enumerate(itr):

        # Extract specific signals
        ecg_data, ecg_freq = [(v, k[1]/10**9) for
                              k, v in w.signals.items() if 'ECG' in k[0]][0]
        ppg_data, ppg_freq = [(v, k[1]/10**9) for
                              k, v in w.signals.items() if 'PULS' in k[0]][0]

        ecg_data['times'] = ecg_data['times'] / 10**9
        ppg_data['times'] = ppg_data['times'] / 10**9

        try:
            ecg_peaks, ppg_peaks, idx_ecg, idx_ppg, m_peaks, pats = align_peaks(
                ecg_data, ecg_freq, ppg_data, ppg_freq)
        except Exception as e:
            # Skip to next window if error extracting PATs
            print(f"Error: {e}")
            continue

        plot_pat(ecg_data, ecg_peaks, ppg_data, ppg_peaks,
                 idx_ecg, idx_ppg, m_peaks, pats,
                 patient_id=w.patient_id, device_id=dev, show=False, save=True)

        clean_pats = pats[(pats > 0) & (pats < 3)]
        plot_pat_hist(clean_pats, patient_id=w.patient_id, device_id=dev,
                      show=False, save=True)

        # Only process if signals are valid
        # if (dv.valid_ecg(ecg['values'], ecg_freq) and
        #         dv.valid_ppg(ppg['values'])):

        if w.patient_id not in p.keys():
            p[w.patient_id] = {"pat": [], "time": []}

        p[w.patient_id]["pat"].append(np.array(pats))
        p[w.patient_id]["time"].append(np.array(ecg_data['times']))

        # np.save(f"raw_data/data_{i}_hourly.npy", w)

    # Flatten all the patient data
    for k, v in p.items():
        p[k]["pat"] = np.concatenate(v["pat"])
        p[k]["time"] = np.concatenate(v["time"])

    return p


def process(dataset_location, device,
            window_size_nano=3600*(10**9), gap_tol_nano=1*(10**9)):
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
        {'tag': "MDC_PRESS_BLD_ART_ABP",
            'freq_nhz': 125_000_000_000, 'units': "MDC_DIM_MMHG"},
        {'tag': "MDC_ECG_ELEC_POTL_II", 'freq_nhz': 500_000_000_000,
            'units': "MDC_DIM_MILLI_VOLT"},
        {'tag': "MDC_PULS_OXIM_PLETH", 'freq_nhz': 125_000_000_000,
            'units': "MDC_DIM_DIMLESS"},
    ]

    print(f"Building dataset, device: {device}")

    definition = DatasetDefinition.build_from_intervals(
        sdk, "measures", measures=measures,
        device_id_list={device: "all"},
        merge_strategy="intersection",
        gap_tolerance=gap_tol_nano)

    itr = sdk.get_iterator(definition,
                           window_size_nano, window_size_nano,
                           num_windows_prefetch=10,
                           # cached_windows_per_source=1,
                           shuffle=False)

    return process_pat(itr, device)


if __name__ == "__main__":

    local_dataset = "/mnt/datasets/atriumdb_abp_estimation_2024_02_05"
    num_cores = 10

    sdk = AtriumSDK(dataset_location=local_dataset)
    devices = list(sdk.get_all_devices().keys())
    print(f"Devices: {devices}")

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as pp:

        futures = {pp.submit(process, local_dataset, d):
                   d for d in devices}

        results = [f.result() for
                   f in concurrent.futures.as_completed(futures)]

        np.save("raw_data/results.npy", results)
        print(f"Results len {len(results)}")

        pats = np.concatenate([v["pat"] for r in results for k, v in r.items()])
        clean_pats = pats[(pats > 0) & (pats < 3)]

        sns.displot(pats, bins=1000, kde=True)
        plt.suptitle(f"PAT over {len(results)} windows")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency")
        plt.savefig("plots/pat_all_dist")

        sns.displot(clean_pats, bins=1000, kde=True)
        plt.suptitle(f"PAT over {len(results)} windows")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency")
        plt.savefig("plots/pat_all_dist_clean")

    print("Finished processing")

    # These columns are not valid in dataset (for de-identification)
    drop = ['mrn', 'first_name', 'middle_name', 'last_name',
            'last_updated', 'first_seen', 'source_id',
            'height', 'weight'
            ]
