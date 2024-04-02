import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def rpeak_dist(ecg_peaks, freq):
    """
    Plot R-Peak IBI Distribution

    :param ecg_peaks: ECG peaks
    :param freq: Sampling frequency
    """

    diff = np.diff(ecg_peaks)

    plt.hist(diff, bins=100)
    plt.title("R-Peak IBI Distribution")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency")
    plt.show()
    plt.savefig("plots/rpeak_ibi_dist")

    return diff


def plot_pat(
    ecg_data,
    ecg_peak_times,
    ppg_data,
    ppg_peak_times,
    pats,
    show=True,
    save=True,
    patient_id=0,
    device_id=0,
):
    """
    Plot PAT

    :param ecg_data: ECG signal
    :param ecg_peak_times: ECG peaks
    :param ppg_data: PPG signal
    :param ppg_peak_times: PPG peaks
    :param pats: PAT
    """

    # Find indicies from values of times
    idx_ecg = np.nonzero(np.in1d(ecg_data["times"], ecg_peak_times))[0]
    idx_ppg = np.nonzero(np.in1d(ppg_data["times"], ppg_peak_times))[0]

    num_plots = 3
    fig, ax = plt.subplots(num_plots, figsize=(15, 10))

    # Share x-axis for all subplots
    for i in range(num_plots):
        ax[i].sharex(ax[0])

    ax[0].plot(ecg_data["times"], ecg_data["values"])
    ax[0].plot(ecg_data["times"][idx_ecg], ecg_data["values"][idx_ecg], "x")
    ax[0].set_title("ECG")
    ax[0].set_xlabel("Time (s)")

    ax[1].plot(ppg_data["times"], ppg_data["values"])
    ax[1].plot(ppg_data["times"][idx_ppg], ppg_data["values"][idx_ppg], "x")
    ax[1].set_title("PPG")
    ax[1].set_xlabel("Time (s)")

    pat_idx = pats[:, 0].astype(int)
    pat_values = pats[:, 1]

    ax[2].plot(ecg_data["times"][idx_ecg][pat_idx], pat_values, "x")
    ax[2].set_title("PAT")
    ax[2].set_xlabel("Time (s)")
    ax[2].set_ylabel("PAT (s)")
    ax[2].set_ylim(1.0, 1.7)
    ax[2].grid(True)

    plt.tight_layout()

    if save:
        plt.savefig(f"plots/pat/pat_{patient_id}_{device_id}")
    if show:
        plt.show()

    plt.close()


def plot_pat_hist(pats, patient_id=0, device_id=0, show=True, save=True):
    """
    Plot PAT histogram

    :param pats: PAT
    """

    sns.displot(pats, bins=100, kde=True)

    # plt.hist(pats, bins=100)
    plt.title("PAT Distribution")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency")

    print(f"Average PAT: {np.mean(pats)}")
    print(f"Median PAT: {np.median(pats)}")
    print(f"Max PAT: {np.max(pats)}")
    print(f"Min PAT: {np.min(pats)}")
    print(f"STD PAT: {np.std(pats)}")

    if save:
        plt.savefig(f"plots/pat/pat_dist_{patient_id}_{device_id}")
    if show:
        plt.show()

    plt.close()
