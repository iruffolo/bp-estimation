import matplotlib.pyplot as plt
import numpy as np


def plot_waveforms(ecg, ppg, abp, show=False):
    """
    Plot all the raw data waveforms for debugging
    """

    fig, ax = plt.subplots(3, figsize=(15, 10), sharex=True)

    ax[0].plot(ecg["times"], ecg["values"])
    ax[0].set_title("ECG")
    ax[0].set_xlabel("Time (s)")

    ax[1].plot(ppg["times"], ppg["values"])
    ax[1].set_title("PPG")
    ax[1].set_xlabel("Time (s)")

    ax[2].plot(abp["times"], abp["values"])
    ax[2].set_title("ABP")
    ax[2].set_xlabel("Time (s)")

    if show:
        plt.tight_layout()
        plt.show()
        plt.close()

    return fig, ax
