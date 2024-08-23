import numpy as np
from atriumdb import AtriumSDK, DatasetDefinition


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


def make_patient_itr(
    sdk,
    patient_id,
    window_size_nano=60 * 20 * (10**9),
    gap_tol_nano=5 * (10**9),
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
    window_size_nano=60 * 30 * (10**9),
    gap_tol_nano=5 * (10**9),
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
    measure_ids = [2, 3, 4] if measure_ids is None else measure_ids
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


def print_all_measures(sdk):
    """
    Prints all measures in this AtriumDB
    """
    m = sdk.get_all_measures()
    for measure in m:
        print(f"{measure}, {m[measure]}")
