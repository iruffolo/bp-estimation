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
    abp_data=None,
    abp_peak_times=None,
    abp_pats=None,
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

    num_plots = 3 + (abp_data is not None)
    fig, ax = plt.subplots(num_plots, figsize=(15, 10))

    # Share x-axis for all subplots
    for i in range(num_plots):
        ax[i].sharex(ax[0])

    plot_idx = 0
    if abp_data is not None:
        idx_abp = np.nonzero(np.in1d(abp_data["times"], abp_peak_times))[0]
        ax[plot_idx].plot(abp_data["times"], abp_data["values"])
        ax[plot_idx].plot(abp_data["times"][idx_abp], abp_data["values"][idx_abp], "x")
        ax[plot_idx].set_title("ABP")
        ax[plot_idx].set_xlabel("Time (s)")
        plot_idx += 1

    ax[plot_idx].plot(ecg_data["times"], ecg_data["values"])
    ax[plot_idx].plot(ecg_data["times"][idx_ecg], ecg_data["values"][idx_ecg], "x")
    ax[plot_idx].set_title("ECG")
    ax[plot_idx].set_xlabel("Time (s)")
    plot_idx += 1

    ax[plot_idx].plot(ppg_data["times"], ppg_data["values"])
    ax[plot_idx].plot(ppg_data["times"][idx_ppg], ppg_data["values"][idx_ppg], "x")
    ax[plot_idx].set_title("PPG")
    ax[plot_idx].set_xlabel("Time (s)")
    plot_idx += 1

    pat_idx = pats[:, 0].astype(int)
    pat_values = pats[:, 1]
    # abp_pat_idx = abp_pats[:, 0].astype(int)
    # abp_pat_values = abp_pats[:, 1]

    ax[plot_idx].plot(ecg_peak_times[pat_idx], pat_values, ".")
    # ax[plot_idx].plot(ecg_peak_times[abp_pat_idx], abp_pat_values, ".")
    ax[plot_idx].set_title("PAT")
    ax[plot_idx].set_xlabel("Time (s)")
    ax[plot_idx].set_ylabel("PAT (s)")
    ax[plot_idx].set_ylim(0, 2.7)
    ax[plot_idx].grid(True)
    plot_idx += 1

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

    plt.tight_layout()

    if save:
        plt.savefig(f"plots/pat/pat_dist_{patient_id}_{device_id}")
    if show:
        plt.show()

    plt.close()


def plot_waveforms(ecg, ppg, abp, patx, paty, show=False):
    """
    Plot all the raw data waveforms for debugging
    """

    fig, ax = plt.subplots(4, figsize=(15, 10), sharex=True)

    ax[0].plot(ecg["times"], ecg["values"])
    ax[0].set_title("ECG")
    ax[0].set_xlabel("Time (s)")

    ax[1].plot(ppg["times"], ppg["values"])
    ax[1].set_title("PPG")
    ax[1].set_xlabel("Time (s)")

    ax[2].plot(abp["times"], abp["values"])
    ax[2].set_title("ABP")
    ax[2].set_xlabel("Time (s)")

    ax[3].plot(patx, paty, ".")
    ax[3].set_title("PAT")
    ax[3].set_xlabel("Time (s)")

    if show:
        plt.tight_layout()
        plt.show()
        plt.close()

    return fig, ax
