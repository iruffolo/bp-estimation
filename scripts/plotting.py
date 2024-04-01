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
    ecg_peaks,
    ppg_data,
    ppg_peaks,
    idx_ecg,
    idx_ppg,
    m_peaks,
    pats,
    hr,
    show=True,
    save=True,
    patient_id=0,
    device_id=0,
):
    """
    Plot PAT

    :param ecg_data: ECG signal
    :param ecg_peaks: ECG peaks
    :param ppg_data: PPG signal
    :param ppg_peaks: PPG peaks
    :param idx_ecg: ECG peak indices
    :param idx_ppg: PPG peak indices
    :param m_peaks: Matching peaks
    :param pats: PAT
    """

    # Find indicies from values of times
    # np.nonzero(np.in1d(A,B))[0]

    num_plots = 7
    fig, ax = plt.subplots(num_plots, figsize=(25, 20))

    # Share x-axis for all subplots
    for i in range(num_plots):
        ax[i].sharex(ax[0])

    ax[0].plot(ecg_data["times"], ecg_data["values"])
    ax[0].plot(ecg_peaks, ecg_data["values"][idx_ecg], "x")
    ax[0].set_title("ECG")
    ax[0].set_xlabel("Time (s)")

    ax[1].plot(ppg_data["times"], ppg_data["values"])
    ax[1].plot(ppg_peaks, ppg_data["values"][idx_ppg], "x")
    ax[1].set_title("PPG")
    ax[1].set_xlabel("Time (s)")

    ax[2].plot(ecg_peaks[: m_peaks.size], m_peaks, "x")
    ax[2].set_title("Distance to matching PPG Peak")
    ax[2].set_xlabel("Time (s)")
    ax[2].set_ylabel("Num Peaks Apart")
    ax[2].yaxis.grid(True)

    ax[3].plot(ecg_peaks[: pats.size], pats, "x")
    ax[3].set_title("PAT")
    ax[3].set_xlabel("Time (s)")
    ax[3].set_ylabel("PAT (s)")
    ax[3].yaxis.grid(True)

    ax[4].plot(ecg_peaks[: pats.size], pats, "x")
    ax[4].set_title("PAT (Zoomed)")
    ax[4].set_xlabel("Time (s)")
    ax[4].set_ylabel("PAT (s)")
    ax[4].set_ylim(1.2, 1.8)
    ax[4].grid(True)

    ax[5].plot(ppg_peaks[:-2], np.diff(hr))
    ax[5].set_title("Change in HR")

    quality = np.absolute(np.diff(hr)) < 20
    ax[6].plot(ppg_peaks[:-2], quality)
    ax[6].set_title("Signal Quality Pass")

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
