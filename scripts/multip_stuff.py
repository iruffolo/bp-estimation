from atriumdb import (T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO,
                      T_TYPE_TIMESTAMP_ARRAY_INT64_NANO, V_TYPE_DELTA_INT64,
                      V_TYPE_DOUBLE, V_TYPE_INT64, AtriumSDK,
                      DatasetDefinition)
from atriumdb.transfer.adb.devices import transfer_devices
from atriumdb.transfer.adb.measures import transfer_measures
from atriumdb.transfer.adb.patients import transfer_patient_info


def write_to_sdk(sdk_hrv, hrv_time_parameters, hrv_freq_parameters, window_data, device_id, data_dict):
    freq_nHz = (10 ** 18) // int(window_data['slide_s'] * (10 ** 9))
    freq_units = "nHz"

    if data_dict['time'].size == 0:
        return
    sorted_times, sorted_inds = np.unique(data_dict['time'], return_index=True)

    for param, hrv_array in data_dict.items():
        if param in hrv_time_parameters:
            units = "ms" if param in ["meanRR", "SDNN", "RMSSD"] else '%'
        elif param in hrv_freq_parameters:
            units = 'ms^2' if param in ['HRV_LF', 'HRV_HF'] else ''
        else:
            continue

        if hrv_array.size == 0:
            continue

        measure_id = sdk_hrv.get_measure_id(measure_tag=param, freq=freq_nHz, freq_units=freq_units, units=units)
        if measure_id is None:
            measure_id = sdk_hrv.insert_measure(measure_tag=param, freq=freq_nHz, freq_units=freq_units, units=units)

        raw_t_t = T_TYPE_TIMESTAMP_ARRAY_INT64_NANO
        encoded_t_t = T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO

        # Determine the raw and encoded value types based on the dtype of value_data
        if np.issubdtype(hrv_array.dtype, np.integer):
            raw_v_t = V_TYPE_INT64
            encoded_v_t = V_TYPE_DELTA_INT64
        else:
            raw_v_t = V_TYPE_DOUBLE
            encoded_v_t = V_TYPE_DOUBLE

        sdk_hrv.write_data(measure_id, device_id, sorted_times, hrv_array[sorted_inds], freq_nHz,
                           int(sorted_times[0]), raw_time_type=raw_t_t,
                           raw_value_type=raw_v_t, encoded_time_type=encoded_t_t, encoded_value_type=encoded_v_t,
                           scale_m=None, scale_b=None, interval_index_mode="fast", gap_tolerance=2 * int(window_data['slide_s'] * (10 ** 9)))


def worker(device_id, queue):
    sdk = AtriumSDK(dataset_location="/mnt/datasets/ecg_copy_2024_07_16")

    device_ids = {device_id: "all"}

    window_data = {'duration_s': 64, 'slide_s': 64, 'gap_tolerance_s': 5}

    measures = [
        {'tag': "MDC_ECG_ELEC_POTL_II", 'freq_nhz': 500000000000, 'units': "MDC_DIM_MILLI_VOLT"},
    ]
    measures_tuple = (measures[0]['tag'], int(measures[0]['freq_nhz'] / (10 ** 9)),
                      measures[0]['units'])
    measure_id = sdk.get_measure_id(measures[0]['tag'], measures[0]['freq_nhz'], measures[0]['units'])

    definition = DatasetDefinition(measures=measures, device_ids=device_ids)
    iterator = sdk.get_iterator(definition, window_data['duration_s'], window_data['slide_s'],
                                time_units="s", gap_tolerance=window_data['gap_tolerance_s'],
                                num_windows_prefetch=100)

    max_saved_windows = 1_000_000

    data_dict = {param: [] for param in
                 ['meanRR', 'SDNN', 'RMSSD', 'pNN50']}
    data_dict['time'] = []

    for window in iterator:
        ecg = window.signals[measures_tuple]
        ecg_times_ms = ecg["times"] / (10 ** 6)
        ecg_values = ecg["values"]
        ecg_freq_hz = measures_tuple[1]
        device_id = window.device_id

        peak_indices = neurokit_rpeak_detect_fast(ecg_values, ecg_freq_hz)

        if len(peak_indices) < 10:
            continue

        rri_ms = rri_calculator(peak_indices[1:-2], ecg_times_ms)

        hrv_time = hrv_time_calculation(rri_ms, window.patient_id, device_id, window.start_time,
                                        ecg_times_ms[-1] * 10 ** 6, hrv_time_parameters)

        # hrv_freq = hrv_freq_calculator(peak_indices, ecg_freq_hz, window.patient_id, device_id,
        #                                window.start_time, ecg_times_ms[-1] * 10 ** 6, hrv_freq_parameters)

        skip, reason = skip_window(rri_ms, hrv_time)
        if skip:
            continue

        data_dict['meanRR'].append(hrv_time[0])
        data_dict['SDNN'].append(hrv_time[1])
        data_dict['RMSSD'].append(hrv_time[2])
        data_dict['pNN50'].append(hrv_time[3])
        # data_dict['HRV_LF'].append(hrv_freq[0])
        # data_dict['HRV_HF'].append(hrv_freq[1])
        # data_dict['HRV_LFHF'].append(hrv_freq[2])

        data_dict['time'].append(window.start_time)

        # Periodically push data to the main process
        if len(data_dict['meanRR']) >= max_saved_windows:
            queue.put((data_dict, len(data_dict['meanRR'])))
            data_dict = {param: [] for param in
                         ['meanRR', 'SDNN', 'RMSSD', 'pNN50']}
            data_dict['time'] = []

    # Push remaining data if any
    if data_dict['meanRR']:
        queue.put((data_dict, len(data_dict['meanRR'])))

    # Indicate this worker is done
    queue.put(None)


def main():
    sdk = AtriumSDK(dataset_location="/mnt/datasets/ecg_copy_2024_07_16")

    window_data = {'duration_s': 64, 'slide_s': 64, 'gap_tolerance_s': 5}

    measures = [
        {'tag': "MDC_ECG_ELEC_POTL_II", 'freq_nhz': 500000000000, 'units': "MDC_DIM_MILLI_VOLT"},
    ]
    measures_tuple = (measures[0]['tag'], int(measures[0]['freq_nhz'] / (10 ** 9)),
                      measures[0]['units'])
    measure_id = sdk.get_measure_id(measures[0]['tag'], measures[0]['freq_nhz'], measures[0]['units'])

    src_device_id_list = list(range(74, 116))
    src_measure_id_list = [measure_id]
    src_patient_id_list = list(sdk.get_all_patients().keys())

    hrv_dataset_location = f"new_HRV_dataset_window_{window_data['duration_s']}s_slide_{window_data['slide_s']}s"

    sdk_hrv = AtriumSDK.create_dataset(dataset_location=hrv_dataset_location, database_type='sqlite')

    transfer_devices(sdk, sdk_hrv, device_id_list=src_device_id_list)
    print("devices transferred")
    _ = transfer_patient_info(sdk, sdk_hrv, patient_id_list=src_patient_id_list, deidentify=True,
                              patient_info_to_transfer=["dob", "gender"],
                              deidentification_functions={"dob": convert_to_month_start_ns})
    print("patients transferred")

    left_dev_id, right_dev_id = 74, 116
    queues = [(dev_id, mp.Queue()) for dev_id in range(left_dev_id, right_dev_id)]
    processes = [mp.Process(target=worker, args=(dev_id, q)) for dev_id, q in queues]

    total_windows = 0
    print("Estimating total windows...")
    for dev_id in range(left_dev_id, right_dev_id):
        device_ids = {dev_id: "all"}
        definition = DatasetDefinition(measures=measures, device_ids=device_ids)
        iterator = sdk.get_iterator(definition, window_data['duration_s'], window_data['slide_s'],
                                    time_units="s", gap_tolerance=window_data['gap_tolerance_s'],
                                    num_windows_prefetch=100)
        total_windows += iterator._length

    # Progress bar
    pbar = tqdm(total=total_windows, desc='Processing Windows')

    max_workers = 100
    p_i = 0
    for p in processes:
        if p_i >= max_workers:
            break
        p.start()
        p_i += 1

    # Main process to gather results
    active_workers = p_i
    while active_workers > 0:
        for device_id, q in queues:
            if not q.empty():
                data = q.get()
                if data is None:
                    active_workers -= 1
                    if p_i < len(processes):
                        processes[p_i].start()
                        p_i += 1
                        active_workers += 1
                else:
                    # Write data to dest database
                    data_dict, num_windows = data
                    data_dict = {key: np.array(value) for key, value in data_dict.items()}
                    write_to_sdk(sdk_hrv, hrv_time_parameters, hrv_freq_parameters, window_data, device_id,
                                 data_dict)
                    pbar.update(num_windows)

    for p in processes:
        p.join()
    pbar.close()
