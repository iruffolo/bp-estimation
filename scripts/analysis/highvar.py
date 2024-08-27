import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from atriumdb import AtriumSDK, DatasetDefinition
from pat import calclulate_pat
from sawtooth import fit_sawtooth


def setup_db(local_dataset, patient_id):
    """
    Setup function for dataset iterator
    """

    measures = [
        {
            "tag": "MDC_PRESS_BLD_ART_ABP",
            "freq_nhz": 125_000_000_000,
            "units": "MDC_DIM_MMHG",
        },
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

    sdk = AtriumSDK(dataset_location=local_dataset)

    window_size_nano = 60 * 10 * (10**9)
    gap_tol_nano = 0.5 * (10**9)

    definition = DatasetDefinition.build_from_intervals(
        sdk,
        "measures",
        measures=measures,
        patient_id_list={patient_id: "all"},
        merge_strategy="intersection",
        gap_tolerance=gap_tol_nano,
    )

    itr = sdk.get_iterator(
        definition,
        window_size_nano,
        window_size_nano,
        num_windows_prefetch=1000,
        # cached_windows_per_source=1,
        shuffle=False,
    )

    return itr


if __name__ == "__main__":

    # Patient IDs
    pat_ids = [13163, 12335, 16122, 10592]

    chosen_one = pat_ids[3]

    itr = setup_db("/mnt/datasets/atriumdb_abp_estimation_2024_02_05", chosen_one)

    timestamps = []
    pat = []
    sawtooth = []
    corrected = []
    sawtooth_A = []
    sawtooth_freq = []
    sawtooth_offset = []

    num_plots = 5
    fig, ax = plt.subplots(num_plots, figsize=(15, 10))

    # Share x-axis for all subplots
    for i in range(num_plots):
        ax[i].sharex(ax[0])

    for i, w in enumerate(itr):
        print(f"Processing window {i}... Patient {w.patient_id}")

        abp_data, abp_freq = [(v, k[1]) for k, v in w.signals.items() if "ABP" in k[0]][
            0
        ]
        ecg_data, ecg_freq = [(v, k[1]) for k, v in w.signals.items() if "ECG" in k[0]][
            0
        ]
        ppg_data, ppg_freq = [
            (v, k[1]) for k, v in w.signals.items() if "PULS" in k[0]
        ][0]

        ecg_data["times"] = ecg_data["times"] / 10**9
        ppg_data["times"] = ppg_data["times"] / 10**9
        abp_data["times"] = abp_data["times"] / 10**9

        try:
            pats, ecg_peak_times, ppg_peak_times = calclulate_pat(
                ecg_data,
                ecg_freq,
                ppg_data,
                ppg_freq,
            )

            if len(pats) == 0:
                continue

            # Find indicies from values of times
            idx_ecg = np.nonzero(np.in1d(ecg_data["times"], ecg_peak_times))[0]
            idx_ppg = np.nonzero(np.in1d(ppg_data["times"], ppg_peak_times))[0]

            pat_idx = pats[:, 0].astype(int)
            pat_values = pats[:, 1]
            pat_x = ecg_data["times"][idx_ecg][pat_idx]

            # Get Sawtooth and plot
            st, params = fit_sawtooth(pat_x, pat_values, False)

            corrected_pat = pat_values - st + params[2]

            # Store a bunch of values for plotting
            timestamps.append(pat_x)
            pat.append(pats)
            sawtooth.append(st)
            corrected.append(corrected_pat)

            for i in range(len(st)):
                sawtooth_A.append(params[0])
                sawtooth_freq.append(params[1])
                sawtooth_offset.append(params[2])

            ax[0].plot(abp_data["times"], abp_data["values"], "b")
            ax[1].plot(ecg_data["times"], ecg_data["values"], "b")
            ax[1].plot(ecg_data["times"][idx_ecg], ecg_data["values"][idx_ecg], "rx")
            ax[2].plot(ppg_data["times"], ppg_data["values"], "b")
            ax[2].plot(ppg_data["times"][idx_ppg], ppg_data["values"][idx_ppg], "rx")

            ax[3].plot(pat_x, st)
            ax[3].plot(pat_x, pat_values, "b.")
            ax[4].plot(pat_x, corrected_pat, "r.")

        except Exception as e:
            print(e)
            continue

    ax[1].set_title("ABP")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_title("ECG")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylim(-5.0, 5.0)
    ax[2].set_title("PPG")
    ax[2].set_xlabel("Time (s)")
    ax[3].set_title("PAT")
    ax[3].set_xlabel("Time (s)")
    ax[3].set_ylabel("Time (s)")
    ax[3].set_ylim(1.0, 1.7)
    ax[3].grid(True)

    ax[4].set_title("Corrected PAT")
    ax[4].set_xlabel("Time (s)")
    ax[4].set_ylabel("Time (s)")
    ax[4].set_ylim(1.0, 1.7)
    ax[4].grid(True)
    ax[4].sharey(ax[3])

    plt.suptitle(f"Patient {chosen_one}")
    plt.tight_layout()
    plt.show()

    # Flatten lists
    pat = np.array([y[1] for x in pat for y in x])
    corrected = np.array([y for x in corrected for y in x])
    sawtooth = np.array([y for x in sawtooth for y in x])
    timestamps = np.array([y for x in timestamps for y in x])
    sawtooth_A = np.array(sawtooth_A)
    sawtooth_freq = np.array(sawtooth_freq)
    sawtooth_offset = np.array(sawtooth_offset)

    print(pat.shape, corrected.shape, sawtooth.shape, timestamps.shape)

    # Convert to pandas
    df = pd.DataFrame(
        {
            "time": timestamps,
            "pat": pat,
            "sawtooth": sawtooth,
            "corrected": corrected,
            "sawtooth_A": sawtooth_A,
            "sawtooth_freq": sawtooth_freq,
            "sawtooth_offset": sawtooth_offset,
        }
    )

    print(df.head())
    # Save to csv
    df.to_csv(f"pats.csv", index=False)

    sns.displot(pat, kde=True)
    plt.show()
