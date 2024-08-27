import matplotlib.pyplot as plt
import numpy as np


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
