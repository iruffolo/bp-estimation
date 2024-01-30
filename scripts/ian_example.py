import numpy as np
from tqdm import tqdm

from atriumdb import AtriumSDK, DatasetDefinition


def main():
    local_dataset_location = "/mnt/datasets/atriumdb_abp_estimation_one_device_v2"

    sdk = AtriumSDK(dataset_location=local_dataset_location)

    measures = [
        {'tag': "MDC_PRESS_BLD_ART_ABP", 'freq_nhz': 125000000000, 'units': "MDC_DIM_MMHG"},
        {'tag': "MDC_ECG_ELEC_POTL_II", 'freq_nhz': 500000000000, 'units': "MDC_DIM_MILLI_VOLT"},
        {'tag': "MDC_PULS_OXIM_PLETH", 'freq_nhz': 125000000000, 'units': "MDC_DIM_DIMLESS"},
    ]

    print("Building dataset")
    gap_tolerance = 60 * (10 ** 9)  # 1 minute in nanoseconds
    definition = DatasetDefinition.build_from_intervals(
        sdk, "measures", measures=measures,
        device_id_list="all", merge_strategy="intersection", gap_tolerance=gap_tolerance)

    print("Saving dataset")
    definition.save("ian_example_definition.yaml", force=True)

    window_size_nano = 5 * (10 ** 9)
    print("Validating dataset")
    iterator = sdk.get_iterator(
        definition, window_size_nano, window_size_nano, num_windows_prefetch=100_000,
        cached_windows_per_source=1000, shuffle=True)
    print("Loading first cache")

    for window_i, window in enumerate(iterator):
        print()
        print(window.start_time)
        print(window.device_id)
        print(window.patient_id)
        for (measure_tag, measure_freq_nhz, measure_units), signal_dict in window.signals.items():
            print(measure_tag, measure_freq_nhz, measure_units, signal_dict['measure_id'])
            print('times', signal_dict['times'])
            print('values', signal_dict['values'])
            print('expected_count', signal_dict['expected_count'])
            print('actual_count', signal_dict['actual_count'])


if __name__ == "__main__":
    main()
