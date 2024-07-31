from atriumdb import AtriumSDK, DatasetDefinition
from pat import calclulate_pat

if __name__ == "__main__":

    local_dataset = "/mnt/datasets/ians_data_2024_06_12"
    sdk = AtriumSDK(dataset_location=local_dataset)

    measures = [
        # {
        #     "tag": "MDC_PRESS_BLD_ART_ABP",
        #     "freq_nhz": 125_000_000_000,
        #     "units": "MDC_DIM_MMHG",
        # },
        {
            "tag": "MDC_ECG_ELEC_POTL_II",
            "freq_nhz": 500_000_000_000,
            "units": "MDC_DIM_MILLI_VOLT",
        },
        {
            "tag": "MDC_PULS_OXIM_PLETH",
            "freq_nhz": 125_000_000_000,
            "units": "MDC_DIM_DIMLESS",
        },
    ]

    definition = DatasetDefinition.build_from_intervals(
        sdk,
        "measures",
        measures=measures,
        device_id_list={80: "all"},
        merge_strategy="intersection",
        gap_tolerance=0.5 * (10**9),
    )

    window_size_nano = 60 * 10 * (10**9)
    itr = sdk.get_iterator(
        definition,
        window_size_nano,
        window_size_nano,
        num_windows_prefetch=100,
        # cached_windows_per_source=10,
        # shuffle=True,
    )

    for w in itr:

        ecg_data, ecg_freq = [(v, k[1]) for k, v in w.signals.items() if "ECG" in k[0]][
            0
        ]
        ppg_data, ppg_freq = [
            (v, k[1]) for k, v in w.signals.items() if "PULS" in k[0]
        ][0]

        ecg_data["times"] = ecg_data["times"] / 10**9
        ppg_data["times"] = ppg_data["times"] / 10**9

        print(f"Freqs: {ecg_freq}, {ppg_freq}")

        pats, ecg_peak_times, ppg_peak_times = calclulate_pat(
            ecg_data, ecg_freq, ppg_data, ppg_freq
        )
        print("Finished calculating PAT...")

        from plotting import plot_pat

        plot_pat(
            ecg_data,
            ecg_peak_times,
            ppg_data,
            ppg_peak_times,
            pats,
        )
