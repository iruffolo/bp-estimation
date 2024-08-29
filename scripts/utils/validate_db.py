from atriumdb import AtriumSDK, DatasetDefinition, transfer_data
from atriumdb.intervals import Intervals
from atriumdb.intervals.union import intervals_union_list


def validate_atriumdb(dataset, gap_tol=1):
    """
    Checks amonut of available data in dataset based on specified measures
    """

    sdk = AtriumSDK(dataset_location=dataset)

    # ABP measure ids
    abp_ids = [4, 22]  # art_abp and art
    sys_ids = [15, 29]  # art_apb_sys and art_sys

    # ECG/PPG measure ids
    ecg_id = 3
    ppg_id = 2

    # ECG derived HR measure id
    hr_id = 11

    total_raw_waveform_time_hours = 0
    total_derived_time_hours = 0
    total_intersection_hours = 0

    gap_tol_nano = gap_tol * (10**9)

    for device in range(74, 115):
        abp_intervals = Intervals(
            intervals_union_list(
                [
                    sdk.get_interval_array(
                        id, device_id=device, gap_tolerance_nano=gap_tol_nano
                    )
                    for id in abp_ids
                ]
            )
        )
        sys_intervals = Intervals(
            intervals_union_list(
                [
                    sdk.get_interval_array(
                        id, device_id=device, gap_tolerance_nano=gap_tol_nano
                    )
                    for id in sys_ids
                ]
            )
        )
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
        hr_intervals = Intervals(
            sdk.get_interval_array(
                hr_id, device_id=device, gap_tolerance_nano=gap_tol_nano
            )
        )

        total_waveform_intervals = ecg_intervals.intersection(
            ppg_intervals
        ).intersection(abp_intervals)

        total_derived_intervals = hr_intervals.intersection(sys_intervals)

        total_raw_waveform_time_hours += total_waveform_intervals.duration() / (
            60 * 60 * (10**9)
        )
        total_derived_time_hours += total_derived_intervals.duration() / (
            60 * 60 * (10**9)
        )
        total_intersection_hours += total_waveform_intervals.intersection(
            total_derived_intervals
        ).duration() / (60 * 60 * (10**9))

    print(f"total_raw_waveform_time_hours: {total_raw_waveform_time_hours}")
    print(f"total_sys_time_hours: {total_derived_time_hours}")
    print(f"total_intersection_hours: {total_intersection_hours}")


if __name__ == "__main__":
    validate_atriumdb("/mnt/datasets/ian_dataset_2024_08_15/", 5 * 60)
