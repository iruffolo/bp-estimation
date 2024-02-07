# import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from tqdm import tqdm
from datetime import datetime
from scipy.signal import find_peaks

from atriumdb import AtriumSDK, DatasetDefinition

from data_quality import DataValidator


def age_distribution(date_of_birth, plot=True):
    """
    Converts date of birth to an integer age.

    :param date_of_birth: Pandas df containing patient date of birth epoch time
    in nanoseconds.
    :param plot: Save a histogram plot, defaults to true.

    :return: New series of ages calculate from the date of birth
    """

    age = date_of_birth.apply(
        lambda x: (datetime.now() -
                   datetime.fromtimestamp(x/10**9)).days/365.2425).astype(int)

    # print(df.age.value_counts())
    # df = df.drop(df[df['age'] > 30].index)

    if plot:
        age.hist(bins=age.nunique())
        plt.title("Patient Age Distribution")
        plt.xlabel("Age (years)")
        plt.ylabel("Count")
        plt.savefig('age_dist.png')

    return age


def get_patient_time(patients, measure):
    """
    For each patient, get how much data exists for the specified measure

    :param patients: List of patient ids.
    :param measure: String or id of measure.
    """

    pass


def get_stats(sdk, measures):

    print("Getting Stats")

    # Get all patient data and clean some useless columns
    patients = sdk.get_all_patients()

    df = pd.DataFrame.from_dict(patients, orient='index')
    df = df.drop(columns=['mrn', 'first_name', 'middle_name', 'last_name',
                          'last_updated', 'first_seen', 'source_id'])

    # Get age distribution
    df['age'] = age_distribution(df['dob'])

    x = sdk.get_device_patient_data(patient_id_list=[46580])
    print(x)

    df = pd.DataFrame(x, columns=['dev', 'id', 'start', 'end'])

    # Convert from nano to sec
    df['end'] = df['end'] / 10**9
    df['start'] = df['start'] / 10**9
    df['dur'] = (df['end'] - df['start']) / 60

    m1 = sdk.get_measure_id('MDC_PRESS_BLD_ART_ABP', freq=125000000000)

    device_patients = sdk.get_device_patient_data(device_id_list=[80])

    for (d, p, start, end) in device_patients:
        print(d, p, sdk.measure_device_start_time_exists(m1, d, start))


def main(sdk, measures):

    print("Building dataset")
    gap_tolerance = 60 * (10 ** 9)  # 1 minute in nanoseconds
    definition = DatasetDefinition.build_from_intervals(
        sdk, "measures", measures=measures,
        device_id_list="all", merge_strategy="intersection",
        gap_tolerance=gap_tolerance)

    print("Saving dataset")
    definition.save("ian_example_definition.yaml", force=True)

    window_size_nano = 32 * (10 ** 9)
    print("Validating dataset")
    iterator = sdk.get_iterator(
        definition, window_size_nano, window_size_nano,
        num_windows_prefetch=1_000,
        # cached_windows_per_source=1,
        shuffle=False)
    print("Loading first cache")

    dv = DataValidator()

    for window_i, window in enumerate(iterator):
        print()
        print(window.device_id, window.patient_id)

        # fig, ax = plt.subplots(3, figsize=(25, 15))
        # fig.suptitle(f"Patient {window.patient_id}")

        for i, ((measure_tag, measure_freq_nhz, measure_units), signal_dict) \
                in enumerate(window.signals.items()):

            print(measure_tag, measure_freq_nhz, measure_units)

            if 'ABP' in measure_tag:
                valid, peaks = dv.valid_abp(signal_dict['values'])
            if 'ECG' in measure_tag:
                valid, peaks = dv.valid_ecg(signal_dict['values'])
            if 'PULS' in measure_tag:
                valid, peaks = dv.valid_ppg(signal_dict['values'])

            # ax[i].plot(signal_dict['times'], signal_dict['values'])
            # ax[i].plot(signal_dict['times'][peaks],
            #            signal_dict['values'][peaks], "x")
            # ax[i].set_title(f"Measure {measure_tag}")
            # ax[i].set_xlabel("Time")
            # ax[i].set_ylabel(f"{measure_units}")

        # plt.tight_layout()
        # plt.savefig(f'plots/patient{window.patient_id}_{window_i}.png')
        # plt.close()

    dv.print_stats(save=True)


if __name__ == "__main__":
    local_dataset = "/mnt/datasets/atriumdb_abp_estimation_one_device_v2"
    sdk = AtriumSDK(dataset_location=local_dataset)

    measures = [
        {'tag': "MDC_PRESS_BLD_ART_ABP",
            'freq_nhz': 125_000_000_000, 'units': "MDC_DIM_MMHG"},
        {'tag': "MDC_ECG_ELEC_POTL_II", 'freq_nhz': 500_000_000_000,
            'units': "MDC_DIM_MILLI_VOLT"},
        {'tag': "MDC_PULS_OXIM_PLETH", 'freq_nhz': 125_000_000_000,
            'units': "MDC_DIM_DIMLESS"},
    ]

    # get_stats(sdk, measures)
    main(sdk, measures)
