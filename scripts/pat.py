import matplotlib.pyplot as plt
import numpy as np
import heartpy as hp
# https://github.com/LaussenLabs/consensus_peaks
from consensus_peaks import consensus_detect


def _get_ppg_peaks(signal, freq):
    """
    Get peaks from PPG
    """

    working_data, measures = hp.process(signal, freq)

    return working_data, measures


def _get_ecg_peaks(signal, freq):
    """
    Get peaks from ECG
    """

    c_peaks = consensus_detect(signal, freq)

    return c_peaks


def closest_argmin(x, y):
    x = x[:, None]
    y = y[None, :]

    z = y-x

    z[np.where(z <= 0)] = np.max(z)

    return np.array(z.argmin(axis=-1)).astype(int)


def calculate_pat(ecg, ecg_freq, ppg, ppg_freq):

    fig, ax = plt.subplots(4, figsize=(25, 15))

    ecg_peaks = _get_ecg_peaks(ecg['values'], ecg_freq)[:-1]

    wd, m = _get_ppg_peaks(ppg['values'], ppg_freq)
    # print(f"{wd}")

    ecg_peak_times = ecg['times'][ecg_peaks]
    ppg_peak_times = ppg['times'][wd['peaklist']]

    times = closest_argmin(ecg_peak_times, ppg_peak_times)

    t = ppg['times'][np.array(wd['peaklist'])[times]]
    e = ecg['times'][ecg_peaks]

    pat = (t - e) / 10**6

    # x = np.ones_like(t)*1.1

    # ax[0].plot(abp['times'], abp['values'])
    # ax[0].set_title("ABP")

    # ax[1].plot(ecg['times'], ecg['values'])
    # ax[1].plot(ecg['times'][ecg_peaks],
    #            ecg['values'][ecg_peaks], "x")
    # ax[1].plot(t, x, "x")
    # ax[1].set_title("ECG")
    #
    # ax[2].plot(ppg['times'], ppg['values'])
    # ax[2].plot(ppg['times'][wd['peaklist']],
    #            ppg['values'][wd['peaklist']], "x")
    #
    # ax[3].plot(ecg_peak_times, pat)
    # ax[3].set_title("PAT")
    #
    # plt.savefig("patttt")

    return pat, ecg_peak_times

if __name__ == "__main__":

    print("PTT")
