import bioread
import matplotlib.pyplot as plt
import numpy as np
from pat import calc_pat_abp, calclulate_pat

if __name__ == "__main__":

    path = "/home/iruffolo/dev/bp-estimation/data/biopac/"

    filepath = "lefthand.acq"

    biopac_data = bioread.read(path + filepath)

    ppg = biopac_data.channels[0]
    ecg = biopac_data.channels[1]

    ecg_data = {"times": ecg.time_index, "values": ecg.data}
    ecg_freq = ecg.samples_per_second

    ppg_data = {"times": ppg.time_index, "values": ppg.data}
    ppg_freq = ppg.samples_per_second

    pats, ecg_peak_times, ppg_peak_times = calclulate_pat(
        ecg_data, ecg_freq, ppg_data, ppg_freq, pat_range=0.300
    )

    npats, n_ecg_peaks_times, n_ppg_peak_times = calc_pat_abp(
        ecg_data, ecg_freq, ppg_data, ppg_freq
    )

    # from pat import calc_pat_abp
    # pats, ecg_peak_times, ppg_peak_times = calc_pat_abp(
    # ecg_data, ecg_freq, ppg_data, ppg_freq
    # )

    print(npats)
    print("Finished calculating PAT...")

    # from plotting import plot_pat, plot_pat_hist
    # plot_pat(ecg_data, ecg_peak_times, ppg_data, ppg_peak_times, pats)
    # plot_pat_hist(pats[:, 1])

    # Create plot
    fig, ax = plt.subplots(4, 1)
    for a in ax:
        a.sharex(ax[0])

    ax[0].plot(ecg_data["times"], ecg_data["values"])
    ax[0].set_title("ECG")
    ax[0].set_xlabel("Time (s)")
    ax[1].plot(ppg_data["times"], ppg_data["values"])
    ax[1].set_title("PPG")
    ax[1].set_xlabel("Time (s)")

    pat_idx = pats[:, 0].astype(int)
    pat_values = pats[:, 1]
    ax[2].plot(ecg_peak_times[pat_idx], pat_values, ".")

    n_pat_idx = npats[:, 0].astype(int)
    n_pat_values = npats[:, 1]

    ax[3].plot(ecg_peak_times[:-1], n_pat_values, ".")

    plt.show()
