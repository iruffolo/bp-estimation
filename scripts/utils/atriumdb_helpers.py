import numpy as np
from atriumdb import AtriumSDK, DatasetDefinition
from atriumdb.intervals import Intervals
from atriumdb.intervals.union import intervals_union_list


def get_all_patient_data(sdk, patient_id):
    """
    Get patient data from AtriumDB

    :param sdk: AtriumSDK instance
    :param patient_id: Patient ID

    :return: Patient data
    """

    patient_data = sdk.get_patient_info(patient_id)

    itr = make_patient_itr(sdk, patient_id)

    sbps = {"times": [], "values": []}

    for i, w in enumerate(itr):

        for k, v in w.signals.items():

            sbp = [v for k, _ in w.signals.items() if "SYS" in k[0]][0]

            sbps["times"].append(sbp["times"] / 10**9)
            sbps["values"].append(sbp["values"])

    sbps["times"] = np.array(sbps["times"]).flatten()
    sbps["values"] = np.array(sbps["values"]).flatten()

    return sbps


def get_patient_data(sdk, patient_id):
    """
    Gets all patient data across devices
    """

    data = sdk.get_device_patient_data(patient_id_list=[patient_id])

    measure_ids = [2, 3, 4, 14, 15, 16, 28, 29, 30]

    for d in data:

        for m in measure_ids:
            interval = sdk.get_interval_array(
                measure_id=m, device_id=d[0], start=d[2], end=d[3]
            )

            print(d[0], m, interval)

    print(data)


def get_ppg_ecg_intervals_device(sdk, dev, gap_tol_nano):

    ecg_id = 3
    ppg_id = 2

    ecg_intervals = Intervals(
        sdk.get_interval_array(
            ecg_id,
            device_id=dev,
            gap_tolerance_nano=gap_tol_nano,
        )
    )
    ppg_intervals = Intervals(
        sdk.get_interval_array(
            ppg_id,
            device_id=dev,
            gap_tolerance_nano=gap_tol_nano,
        )
    )
    total_waveform_intervals = ecg_intervals.intersection(ppg_intervals)

    return total_waveform_intervals.interval_arr


def get_ppg_ecg_intervals_patient(sdk, pid, dev, gap_tol_nano):
    ecg_id = 3
    ppg_id = 2

    # Get device mapping for patient
    dev_map = sdk.get_device_patient_data(patient_id_list=[pid])

    # Only get data corresponding to specific device
    start = [d[2] for d in dev_map if d[0] == dev][0]
    end = [d[3] for d in dev_map if d[0] == dev][0]

    ecg_intervals = Intervals(
        sdk.get_interval_array(
            ecg_id,
            patient_id=pid,
            gap_tolerance_nano=gap_tol_nano,
            start=start,
            end=end,
        )
    )
    ppg_intervals = Intervals(
        sdk.get_interval_array(
            ppg_id,
            patient_id=pid,
            gap_tolerance_nano=gap_tol_nano,
            start=start,
            end=end,
        )
    )
    total_waveform_intervals = ecg_intervals.intersection(ppg_intervals)

    return total_waveform_intervals


def get_ppg_ecg_data(sdk, pid, dev, gap_tol, window=None):
    """
    Get PPG and ECG data for a specific patient and device
    """

    ecg_id = 3
    ppg_id = 2

    total_waveform_intervals = get_ppg_ecg_intervals_patient(sdk, pid, dev, gap_tol)

    ecg = {"times": np.array([]), "values": np.array([])}
    ppg = {"times": np.array([]), "values": np.array([])}

    for i in total_waveform_intervals.interval_arr:

        # Get ECG and PPG data for all the overlapping intervals

        if window:
            end = i[0] + window
        else:
            end = i[1]

        _, ecg_times, ecg_values = sdk.get_data(
            measure_id=ecg_id, start_time_n=i[0], end_time_n=end, patient_id=pid
        )
        ecg["times"] = np.concatenate((ecg["times"], ecg_times), axis=0)
        ecg["values"] = np.concatenate((ecg["values"], ecg_values), axis=0)

        _, ppg_times, ppg_values = sdk.get_data(
            measure_id=ppg_id, start_time_n=i[0], end_time_n=end, patient_id=pid
        )
        ppg["times"] = np.concatenate((ppg["times"], ppg_times), axis=0)
        ppg["values"] = np.concatenate((ppg["values"], ppg_values), axis=0)

    ecg_info = sdk.get_measure_info(ecg_id)
    ppg_info = sdk.get_measure_info(ppg_id)
    ecg_freq = int(ecg_info["freq_nhz"] / 10**9)
    ppg_freq = int(ppg_info["freq_nhz"] / 10**9)

    # Convert times to seconds
    ppg["times"] = ppg["times"] / 10**9
    ecg["times"] = ecg["times"] / 10**9

    return ppg, ecg, ppg_freq, ecg_freq


def make_patient_itr(
    sdk,
    patient_id,
    window_size_nano=60 * 20,
    gap_tol_nano=5,
):
    """
    Creates new SDK instance and iterator for a specific device

    :param sdk: AtriumSDK instance
    :param device: Device ID
    :param window_size_nano: Window size in nanoseconds
    :param gap_tol_nano: Gap tolerance in nanoseconds

    :return: Dictionary of pulse arrival times for each patient in device
    """

    measure_ids = [14, 15, 16, 28, 29, 30]
    measures = [
        {
            "tag": sdk.get_measure_info(measure_id)["tag"],
            "freq_nhz": sdk.get_measure_info(measure_id)["freq_nhz"],
            "units": sdk.get_measure_info(measure_id)["unit"],
        }
        for measure_id in measure_ids
    ]
    # print(measures)

    definition = DatasetDefinition.build_from_intervals(
        sdk,
        "measures",
        measures=measures,
        patient_id_list={patient_id: "all"},
        merge_strategy="union",
        gap_tolerance=gap_tol_nano,
    )

    itr = sdk.get_iterator(
        definition,
        window_size_nano,
        window_size_nano,
        num_windows_prefetch=10,
        # cached_windows_per_source=20,
        shuffle=False,
    )

    return itr


def make_device_itr(
    sdk,
    device,
    window_size=60 * 30,
    gap_tol=5,
    measure_ids=None,
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
    measure_ids = [2, 3, 4, 7, 15] if measure_ids is None else measure_ids
    measures = [
        {
            "tag": sdk.get_measure_info(measure_id)["tag"],
            "freq_nhz": sdk.get_measure_info(measure_id)["freq_nhz"],
            "units": sdk.get_measure_info(measure_id)["unit"],
        }
        for measure_id in measure_ids
    ]

    definition = DatasetDefinition.build_from_intervals(
        sdk,
        "measures",
        measures=measures,
        device_id_list={device: "all"},
        merge_strategy="intersection",
        gap_tolerance=gap_tol * (10**9),
    )

    itr = sdk.get_iterator(
        definition,
        window_size * (10**9),
        window_size * (10**9),
        num_windows_prefetch=50,
        # cached_windows_per_source=20,
        shuffle=False,
    )

    return itr


def make_device_itr_ecg_ppg(
    sdk,
    window_size,
    gap_tol,
    device=None,
    prefetch=10,
    cache=1,
    shuffle=False,
    start_nano=None,
    end_nano=None,
):
    """
    Creates new SDK instance and iterator for a specific device

    :param sdk: AtriumSDK instance
    :param device: Device ID
    :param window_size: Window size in nanoseconds
    :param gap_tol: Gap tolerance in nanoseconds

    :return: Dictionary of pulse arrival times for each patient in device
    """

    print(f"Building dataset, device: {device}")

    # ECG/PPG measure ids
    ecg_id = 3
    ppg_id = 2

    # Create measures list
    measure_ids = [ecg_id, ppg_id]
    measures = [
        {
            "tag": sdk.get_measure_info(measure_id)["tag"],
            "freq_nhz": sdk.get_measure_info(measure_id)["freq_nhz"],
            "units": sdk.get_measure_info(measure_id)["unit"],
        }
        for measure_id in measure_ids
    ]

    gap_tol_nano = gap_tol * (10**9)

    ecg_intervals = Intervals(
        sdk.get_interval_array(
            ecg_id,
            device_id=device,
            gap_tolerance_nano=gap_tol_nano,
            start=start_nano,
            end=end_nano,
        )
    )
    ppg_intervals = Intervals(
        sdk.get_interval_array(
            ppg_id,
            device_id=device,
            gap_tolerance_nano=gap_tol_nano,
            start=start_nano,
            end=end_nano,
        )
    )

    total_waveform_intervals = ecg_intervals.intersection(ppg_intervals)

    device_ids = {
        device: [
            {"start": int(start), "end": int(end)}
            for start, end in total_waveform_intervals.interval_arr
        ]
    }

    definition = DatasetDefinition(measures=measures, device_ids=device_ids)
    sdk.load_definition(definition)

    itr = sdk.get_iterator(
        definition,
        window_duration=window_size,
        window_slide=window_size,
        gap_tolerance=gap_tol,
        num_windows_prefetch=prefetch,
        cached_windows_per_source=cache,
        time_units="s",
        shuffle=shuffle,
        iterator_type="lightmapped",
    )

    return itr
    itr = sdk.get_iterator(
        definition,
        window_duration=window_size,
        window_slide=window_size,
        gap_tolerance=gap_tol,
        num_windows_prefetch=prefetch,
        cached_windows_per_source=cache,
        time_units="s",
        shuffle=shuffle,
        iterator_type="lightmapped",
    )

    return itr


def make_device_itr_all_signals(
    sdk,
    window_size,
    gap_tol,
    device=None,
    pid=None,
    prefetch=10,
    shuffle=True,
    start=None,
    end=None,
):
    """
    Creates new SDK instance and iterator for a specific device

    :param sdk: AtriumSDK instance
    :param device: Device ID
    :param window_size: Window size in nanoseconds
    :param gap_tol: Gap tolerance in nanoseconds

    :return: Dictionary of pulse arrival times for each patient in device
    """

    print(f"Building dataset, device: {device}")

    # ABP measure ids
    abp_ids = [4, 22]  # art_abp and art
    sys_ids = [15, 29]  # art_apb_sys and art_sys

    # ECG/PPG measure ids
    ecg_id = 3
    ppg_id = 2

    # ECG derived HR measure id
    hr_id = 11

    # Create measures list
    measure_ids = [*abp_ids, *sys_ids, ecg_id, ppg_id, hr_id]
    measures = [
        {
            "tag": sdk.get_measure_info(measure_id)["tag"],
            "freq_nhz": sdk.get_measure_info(measure_id)["freq_nhz"],
            "units": sdk.get_measure_info(measure_id)["unit"],
        }
        for measure_id in measure_ids
    ]

    gap_tol_nano = gap_tol * (10**9)

    abp_intervals = Intervals(
        intervals_union_list(
            [
                sdk.get_interval_array(
                    id,
                    device_id=device,
                    patient_id=pid,
                    gap_tolerance_nano=gap_tol_nano,
                    start=start,
                    end=end,
                )
                for id in abp_ids
            ]
        )
    )
    sys_intervals = Intervals(
        intervals_union_list(
            [
                sdk.get_interval_array(
                    id,
                    device_id=device,
                    patient_id=pid,
                    gap_tolerance_nano=gap_tol_nano,
                    start=start,
                    end=end,
                )
                for id in sys_ids
            ]
        )
    )
    ecg_intervals = Intervals(
        sdk.get_interval_array(
            ecg_id,
            device_id=device,
            patient_id=pid,
            gap_tolerance_nano=gap_tol_nano,
            start=start,
            end=end,
        )
    )
    ppg_intervals = Intervals(
        sdk.get_interval_array(
            ppg_id,
            device_id=device,
            patient_id=pid,
            gap_tolerance_nano=gap_tol_nano,
            start=start,
            end=end,
        )
    )
    hr_intervals = Intervals(
        sdk.get_interval_array(
            hr_id,
            device_id=device,
            patient_id=pid,
            gap_tolerance_nano=gap_tol_nano,
            start=start,
            end=end,
        )
    )

    total_waveform_intervals = ecg_intervals.intersection(ppg_intervals).intersection(
        abp_intervals
    )
    total_derived_intervals = hr_intervals.intersection(sys_intervals)

    device_ids = {
        device: [
            {"start": int(start), "end": int(end)}
            for start, end in total_waveform_intervals.intersection(
                total_derived_intervals
            ).interval_arr
        ]
    }
    # print(device_ids)

    definition = DatasetDefinition(measures=measures, device_ids=device_ids)

    itr = sdk.get_iterator(
        definition,
        window_duration=window_size,
        window_slide=window_size,
        gap_tolerance=gap_tol,
        num_windows_prefetch=prefetch,
        time_units="s",
        shuffle=shuffle,
    )

    return itr


def print_all_measures(sdk):
    """
    Prints all measures in this AtriumDB
    """
    m = sdk.get_all_measures()
    for measure in m:
        print(f"{measure}, {m[measure]}")


if __name__ == "__main__":

    sdk = AtriumSDK(dataset_location="/mnt/datasets/ian_dataset_2024_08_15")
    print_all_measures(sdk)

    test_dev = 80
    window_size = 60 * 30  # 30 min
    gap_tol = 5

    itr = make_device_itr_all_signals(sdk, test_dev, window_size, gap_tol)

    for i, w in enumerate(itr):
        print(i, w)
