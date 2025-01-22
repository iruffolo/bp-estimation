import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from atriumdb import AtriumSDK, DatasetDefinition
from pat import calclulate_pat
from utils.atriumdb_helpers import make_device_itr_all_signals


def setup_db(localset, patient_id):
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

    sdk = AtriumSDK(dataset_location=localset)

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
        num_windows_prefetch=100,
        # cached_windows_per_source=1,
        shuffle=True,
    )

    return itr


if __name__ == "__main__":

    # itr = setup_db("/mnt/datasets/ianset_2024_08_26", 13561)

    localset = "/mnt/datasets/ian_dataset_2024_08_26"
    sdk = AtriumSDK(dataset_location=localset)
    pid = 13561
    window_size = 60 * 60  # 60 min
    gap_tol = 30 * 60  # 5 min to reduce overlapping windows with gap tol

    itr = make_device_itr_all_signals(
        sdk,
        window_size,
        gap_tol,
        device=90,
        pid=pid,
        prefetch=10,
        shuffel=False,
    )

    print(len(itr))

    pat = []

    num_plots = 3
    fig, ax = plt.subplots(num_plots, figsize=(15, 10))

    # Share x-axis for all subplots
    for i in range(num_plots):
        ax[i].sharex(ax[0])

    for i, w in enumerate(itr):
        print(f"Processing window {i}... Patient {w.patient_id}")

        # Extract data from window and validate
        for (signal, freq, _), v in w.signals.items():
            # Skip signals without data (ABP, SYS variants)
            if v["actual_count"] == 0:
                continue

            # Convert to s
            v["times"] = v["times"] / (10**9)

            # Extract specific signals
            match signal:
                case signal if "ECG_ELEC" in signal:
                    ecg, ecg_freq = v, freq
                case signal if "PULS_OXIM" in signal:
                    ppg, ppg_freq = v, freq
                case _:
                    pass

        pats, naive_pats, n_corr, ecg_peak_times, ppg_peak_times = calclulate_pat(
            ecg, ecg_freq, ppg, ppg_freq
        )
        print(pats)
        pat.append(pats)

        # Find indicies from values of times
        idx_ecg = np.nonzero(np.in1d(ecg["times"], ecg_peak_times))[0]
        idx_ppg = np.nonzero(np.in1d(ppg["times"], ppg_peak_times))[0]

        ax[0].plot(ecg["times"], ecg["values"], "b")
        ax[0].plot(ecg["times"][idx_ecg], ecg["values"][idx_ecg], "rx")
        ax[1].plot(ppg["times"], ppg["values"], "b")
        ax[1].plot(ppg["times"][idx_ppg], ppg["values"][idx_ppg], "rx")

        ax[2].plot(pats["times"], pats["values"], "b.")

        if i > 10:
            break

    ax[0].set_title("ECG")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylim(-5.0, 5.0)
    ax[1].set_title("PPG")
    ax[1].set_xlabel("Time (s)")
    ax[2].set_title("PAT")
    ax[2].set_xlabel("Time (s)")
    ax[2].set_ylabel("PAT (s)")
    ax[2].set_ylim(0, 2.5)
    ax[2].grid(True)

    plt.tight_layout()
    plt.show()

    sns.displot(pats["values"], kde=True)
    plt.show()
