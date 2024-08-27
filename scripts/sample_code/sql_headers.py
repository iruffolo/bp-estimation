import numpy as np
from atriumdb import AtriumSDK
from atriumdb.adb_functions import (condense_byte_read_list, get_headers,
                                    time_unit_options)


def main():
    local_dataset_location = "/mnt/datasets/atriumdb_abp_estimation_pleth_2024_04_03"

    sdk_2 = AtriumSDK(dataset_location=local_dataset_location)

    measure_id = 3
    device_id = 84

    interval_arr = sdk_2.get_interval_array(measure_id=measure_id, device_id=device_id)
    start, end = interval_arr[0][0], interval_arr[-1][1]

    all_headers = get_headers(sdk_2, measure_id, start, end, device_id)

    for i, header in enumerate(all_headers):
        print(f"New Header {i}")
        print(header.start_n, header.end_n)
        print(header.min, header.max, header.mean, header.std)
        analog_std = (header.scale_m * header.std) + header.scale_b
        print(header.scale_m, header.scale_b, analog_std)


if __name__ == "__main__":
    main()
