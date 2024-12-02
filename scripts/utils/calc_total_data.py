from atriumdb import AtriumSDK
from atriumdb.intervals import Intervals
from atriumdb.intervals.union import intervals_union_list
from atriumdb_helpers import make_device_itr_ecg_ppg


def get_intervals(sdk, window_size, gap_tol, device, prefetch):
    """
    Function to get intervals for ECG and PPG measures
    """

    # ecg/ppg measure ids
    ecg_id = 3
    ppg_id = 2

    # create measures list
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
            ecg_id, device_id=device, gap_tolerance_nano=gap_tol_nano
        )
    )
    ppg_intervals = Intervals(
        sdk.get_interval_array(
            ppg_id, device_id=device, gap_tolerance_nano=gap_tol_nano
        )
    )

    total_waveform_intervals = ecg_intervals.intersection(ppg_intervals)

    device_ids = {
        device: [
            {"start": int(start), "end": int(end)}
            for start, end in total_waveform_intervals.interval_arr
        ]
    }

    return device_ids


if __name__ == "__main__":

    local_dataset = "/mnt/datasets/ian_dataset_2024_08_26"
    sdk = AtriumSDK(dataset_location=local_dataset)

    window_size = 1 * 60 * 60  # 30 min
    gap_tol = 30 * 60  # 30 min to reduce overlapping windows with gap tol

    devices = list(sdk.get_all_devices().keys())

    device = 80

    t_duration = 0
    for d in devices:
        interval = get_intervals(
            sdk,
            window_size,
            gap_tol,
            device=d,
            prefetch=10,
        )

        d_duration = 0
        for i in interval[d]:
            duration_s = (i["end"] - i["start"]) / 10**9

            d_duration += duration_s
        t_duration += d_duration
        print(f"Device {d} duration: {d_duration}")
    print(f"Total duration: {t_duration}s")
    print(f"Total duration: {t_duration/60} min")
    print(f"Total duration: {t_duration/(60*60)} hours")
