import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import concurrent.futures

from atriumdb import AtriumSDK, DatasetDefinition

from data_quality import DataValidator
from pat import calculate_pat

from stats import age_distribution


def process_window(*args):
    """
    Processes window of data with ABP, ECG, PPG

    """
    (window_i, window) = args
    dv = DataValidator()

    fig, ax = plt.subplots(3, figsize=(25, 15))
    fig.suptitle(f"Patient {window.patient_id}")

    valid = True

    for i, ((measure_tag, measure_freq_nhz, measure_units), signal_dict) \
            in enumerate(window.signals.items()):

        print(measure_tag, measure_freq_nhz, measure_units)

        freq_hz = measure_freq_nhz / 10**9

        if 'ABP' in measure_tag:
            valid, peaks = dv.valid_abp(signal_dict['values'])
        if 'ECG' in measure_tag:
            valid, peaks, c_peaks = dv.valid_ecg(
                signal_dict['values'], freq_hz)
            ax[i].plot(signal_dict['times'][c_peaks],
                       signal_dict['values'][c_peaks], "o")
        if 'PULS' in measure_tag:
            valid, peaks = dv.valid_ppg(signal_dict['values'])

        ax[i].plot(signal_dict['times'], signal_dict['values'])
        ax[i].plot(signal_dict['times'][peaks],
                   signal_dict['values'][peaks], "x")
        ax[i].set_title(f"Measure {measure_tag}")
        ax[i].set_xlabel("Time")
        ax[i].set_ylabel(f"{measure_units}")

    dv.print_stats(save=False)
    plt.tight_layout()
    if valid:
        plt.savefig(
            f'plots_valid/patient{window.patient_id}_{window_i}.png')
    else:
        plt.savefig(
            f'plots_invalid/patient{window.patient_id}_{window_i}.png')
    plt.close()


def get_stats(sdk, measures):

    print("Getting Stats")

    # Get all patient data and clean some useless columns
    patients = sdk.get_all_patients()

    df = pd.DataFrame.from_dict(patients, orient='index')

    # These columns are not valid in dataset (for de-identification)
    df = df.drop(columns=['mrn', 'first_name', 'middle_name', 'last_name',
                          'last_updated', 'first_seen', 'source_id'])

    # Get age distribution
    df['age'] = age_distribution(df['dob'], plot=True)


def process_ptt(itr, dev):
    """
    Processing function for dataset iterator
    """

    p = {}
    dv = DataValidator()

    for i, w in enumerate(itr):
        print(f"patient {w.patient_id}")
        if w.patient_id not in p.keys():
            p[w.patient_id] = {"pat": [], "time": []}

        # Extract specific signals
        ecg = [v for k, v in w.signals.items() if 'ECG' in k[0]][0]
        ecg_freq = [k[1] for k, v in w.signals.items() if 'ECG' in k[0]][0]
        ppg = [v for k, v in w.signals.items() if 'PULS' in k[0]][0]
        ppg_freq = [k[1] for k, v in w.signals.items() if 'PULS' in k[0]][0]
        # abp = [v for k, v in w.signals.items() if 'ABP' in k[0]][0]

        if (dv.valid_ecg(ecg['values'], ecg_freq) and
                dv.valid_ppg(ppg['values'])):
            print("Valid")

            pat, t = calculate_pat(ecg, ecg_freq/10**9, ppg, ppg_freq/10**9)

            p[w.patient_id]["pat"].append(pat)
            p[w.patient_id]["time"].append(t)

        else:
            print("Invalid window, skipping")

        break

    for pat in p.keys():
        print(f"Plotting pat {pat}")
        df = pd.DataFrame(np.array(p[pat]["pat"]).flatten(), columns=["pat"])

        sns.displot(data=df, x="pat", kde=True)
        plt.savefig(f"plots/pat/patient{pat}_{dev}")
        plt.close()

        p[pat]["pat"] = df

    return p


def make_patients_dataset(sdk, meas, gap_tol_s=1,
                          pats_per_ds=100, path="datasets/patients"):
    """
    Creates multiple DatasetDefinitions by patient and saves them to folder,
    for faster loading.

    :param sdk: AtriumDB sdk.
    :param meas: List of measures to use in the Dataset.
    :param gap_tol_s: Gap tolerance (missing data) allowed, in seconds.
    :param pats_per_ds: Number of patients to include in each Dataset.
    :param path: Path to folder to save Datasets.

    :return: None
    """

    patients = sdk.get_all_patients()
    ids = {k: 'all' for k in patients.keys()}

    gap_tol_ns = gap_tol_s * (10 ** 9)

    for i in range(0, len(patients), pats_per_ds):

        p = {k: ids[k] for k in list(ids)[i:i+pats_per_ds]}

        print(f"Building dataset, patients: {i}:{i+pats_per_ds}")

        definition = DatasetDefinition.build_from_intervals(
            sdk, "measures", measures=meas,
            patient_id_list=p,
            # device_id_list="all",
            merge_strategy="intersection",
            gap_tolerance=gap_tol_ns)

        definition.save(f"{path}/pats_{i}_{i+pats_per_ds}.yaml", force=True)


def make_devices_dataset(sdk, meas, gap_tol_s=1, path="datasets/devices"):
    """
    Creates multiple DatasetDefinitions by device and saves them to folder,
    for faster loading.

    :param sdk: AtriumDB sdk.
    :param meas: List of measures to use in the Dataset.
    :param gap_tol_s: Gap tolerance (missing data) allowed, in seconds.
    :param path: Path to folder to save Datasets.

    :return: None
    """

    devices = sdk.get_all_devices()

    gap_tol_ns = gap_tol_s * (10 ** 9)

    for dev in devices:

        print(f"Building dataset, device: {dev}")

        definition = DatasetDefinition.build_from_intervals(
            sdk, "measures", measures=meas,
            device_id_list={dev: "all"},
            merge_strategy="intersection",
            gap_tolerance=gap_tol_ns)

        definition.save(f"{path}/dev_{dev}.yaml", force=True)


def process(sdk, device, window_size_nano=32*(10**9), gap_tol_nano=1*(10**9)):
    """
    Creates new SDK instance and iterator for device
    """

    sdk = AtriumSDK(dataset_location=local_dataset)

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
                           num_windows_prefetch=1_000,
                           # cached_windows_per_source=1,
                           shuffle=False)

    return process_ptt(itr, device)


if __name__ == "__main__":

    local_dataset = "/mnt/datasets/atriumdb_abp_estimation_2024_02_05"

    sdk = AtriumSDK(dataset_location=local_dataset)

    devices = list(sdk.get_all_devices().keys())

    # make_devices_dataset(sdk, measures)
    # make_patients_dataset(sdk, measures)

    num_cores = 10

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as pp:

        futures = {pp.submit(process, local_dataset, d):
                   d for d in devices[0:4]}

        x = [p["pat"] for p in concurrent.futures.as_completed(futures)]
        df = pd.concat(x)

        sns.displot(data=df, x="pat", kde=True)
        plt.savefig(f"plots/pat_all")
        plt.close()


    # These columns are not valid in dataset (for de-identification)
    drop = ['mrn', 'first_name', 'middle_name', 'last_name',
            'last_updated', 'first_seen', 'source_id',
            'height', 'weight'
            ]
