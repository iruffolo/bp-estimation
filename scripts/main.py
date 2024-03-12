import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import concurrent.futures

from atriumdb import AtriumSDK, DatasetDefinition
from data_quality import DataValidator
from pat import calculate_pat


def process_pat(itr, dev):
    """
    Processing function for dataset iterator

    :param itr: Dataset iterator
    :param dev: Device ID

    :return: Dictionary of pulse arrival times for each patient in device
    """

    p = {}
    dv = DataValidator()

    for i, w in enumerate(itr):
        print(f"Window {i}: Patient {w.patient_id}")

        if w.patient_id not in p.keys():
            p[w.patient_id] = {"pat": [], "time": [], "abp": []}

        # Extract specific signals
        ecg = [v for k, v in w.signals.items() if 'ECG' in k[0]][0]
        ppg = [v for k, v in w.signals.items() if 'PULS' in k[0]][0]

        ppg_freq = [k[1]
                    for k, v in w.signals.items() if 'PULS' in k[0]][0]/10**9
        ecg_freq = [k[1]
                    for k, v in w.signals.items() if 'ECG' in k[0]][0]/10**9

        # abp = [v for k, v in w.signals.items() if 'ABP' in k[0]][0]
        print(ppg_freq, ecg_freq)
        print(ppg['values'].shape, ecg['values'].shape)

        # Only process if signals are valid
        # if (dv.valid_ecg(ecg['values'], ecg_freq) and
        #         dv.valid_ppg(ppg['values'])):

        # pat, t = calculate_pat(ecg, ecg_freq, ppg, ppg_freq)

        # print(pat)
            # p[w.patient_id]["pat"].append(np.array(pat))
            # p[w.patient_id]["time"].append(np.array(t))
        np.save(f"raw_data/data_{i}_hourly.npy", w)

        break

    # for pat in p.keys():
    #
    #     print(f"Plotting pat {pat}")
    #     data = np.concatenate(p[pat]["pat"]).reshape(-1)
    #     # t = np.concatenate(p[pat]["time"]).reshape(-1)
    #
    #     df = pd.DataFrame(data, columns=["pat"])
    #
    #     sns.displot(data=df, x="pat", kde=True)
    #     plt.suptitle(f"Patient {pat}")
    #     plt.savefig(f"plots/pat/patient{pat}_{dev}.png")
    #     plt.close()

        # fig, ax = plt.subplots(2, figsize=(25, 15))
        # ax[0].plot(abp['times'], abp['values'])
        # ax[0].set_title("ABP")
        #
        # ax[1].plot(ecg_peak_times, pat)
        # ax[1].set_title("PAT")
        #
        # p[pat]["pat"] = df

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
                           num_windows_prefetch=1,
                           cached_windows_per_source=1,
                           shuffle=True)

    return process_pat(itr, device)


if __name__ == "__main__":

    local_dataset = "/mnt/datasets/atriumdb_abp_estimation_2024_02_05"
    num_cores = 2

    sdk = AtriumSDK(dataset_location=local_dataset)
    devices = list(sdk.get_all_devices().keys())
    print(f"Devices: {devices}")

    pat = process(local_dataset, 97)
    print(pat)

    exit()

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as pp:

        futures = {pp.submit(process, local_dataset, 74): 74}
                   # d for d in devices}

        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            dev = futures[future]
            print(f"Device {dev} done")
            print(res)

            # x = [pats["pat"] for patient, pats in res.items()]
            #
            # df = pd.concat(x)
            #
            # sns.displot(data=df, x="pat", kde=True)
            # plt.suptitle(f"PAT over {len(df)} windows (Device {dev})")
            # plt.savefig(f"plots/pat_{dev}")
            #
            # plt.close()

    # These columns are not valid in dataset (for de-identification)
    drop = ['mrn', 'first_name', 'middle_name', 'last_name',
            'last_updated', 'first_seen', 'source_id',
            'height', 'weight'
            ]
