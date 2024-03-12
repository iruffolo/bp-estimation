import matplotlib.pyplot as plt
import numpy as np
import heartpy as hp
from scipy.spatial import distance
# https://github.com/LaussenLabs/consensus_peaks
from consensus_peaks import consensus_detect


def _get_ppg_peaks(signal, freq):
    """
    Get peaks from PPG
    """

    signal = hp.filter_signal(signal, sample_rate=freq,
                              cutoff=40, order=2, filtertype='lowpass')

    working_data, measures = hp.process(signal, freq)

    return working_data['peaklist']


def _get_ecg_peaks(signal, freq):
    """
    Get peaks from ECG
    """

    signal = hp.filter_signal(signal, sample_rate=freq,
                              cutoff=40, order=2, filtertype='lowpass')

    c_peaks = consensus_detect(signal, freq)

    return c_peaks


def closest_argmin(x, y):
    x = x[:, None]
    y = y[None, :]

    z = y-x

    z[np.where(z <= 0)] = np.max(z)

    return np.array(z.argmin(axis=-1)).astype(int)


def get_ecg_signature(ecg, ecg_freq, ppg, ppg_freq, size=10):

    ecg_peaks = _get_ecg_peaks(ecg['values'], ecg_freq)
    ecg_times = ecg['times'][ecg_peaks]
    ecg_diffs = [np.roll(ecg_times, -i) - ecg_times for i in range(size)]
    # Stack so each row represents a peak and its diffs
    ecg_diffs = np.stack(ecg_diffs, axis=1)  # [0]

    # Only keep valid diffs where peaks didnt wrap around
    # ecg_diffs = ecg_diffs[ecg_diffs.min(axis=1) >= 0, :]

    ppg_peaks = _get_ppg_peaks(ppg['values'], ppg_freq)
    ppg_times = ppg['times'][ppg_peaks]
    ppg_diffs = [np.roll(ppg_times, -i) - ppg_times for i in range(size)]
    ppg_diffs = np.stack(ppg_diffs, axis=1)
    # ppg_diffs = ppg_diffs[ppg_diffs.min(axis=1) >= 0, :]

    print(ppg_diffs.shape)

    matching_peaks = list()
    for i in range(ecg_diffs.shape[0]):

        if i > ecg_diffs.shape[0] - size:
            matching_peaks.append(-1)

        else:
            dist = np.array([distance.euclidean(ecg_diffs[i], ppg_diffs[i+j])
                             for j in range(size)])

            matching_peaks.append(np.argmin(dist))

    print(matching_peaks)

    return ecg_peaks, ppg_peaks, matching_peaks


def plot_all(abp, ecg, ppg, save=True, show=True):

    fig, ax = plt.subplots(4, figsize=(25, 15))

    # x = np.ones_like(t)*1.1

    ax[0].plot(abp['times'], abp['values'])
    ax[0].set_title("ABP")

    ax[1].plot(ecg['times'], ecg['values'])
    # ax[1].plot(ecg['times'][ecg_peaks],
    #            ecg['values'][ecg_peaks], "x")

    # ax[1].plot(t, x, "x")
    ax[1].set_title("ECG")

    ax[2].plot(ppg['times'], ppg['values'])
    # ax[2].plot(ppg['times'][wd['peaklist']],
    # ppg['values'][wd['peaklist']], "x")

    # ax[3].plot(ecg_peak_times, pat)
    # ax[3].set_title("PAT")

    if save:
        plt.savefig("patttt")
    if show:
        plt.show()


def calculate_pat(ecg, ecg_freq, ppg, ppg_freq):

    ecg_peaks = _get_ecg_peaks(ecg['values'], ecg_freq)[:-1]

    wd, m = _get_ppg_peaks(ppg['values'], ppg_freq)

    ecg_peak_times = ecg['times'][ecg_peaks]
    ppg_peak_times = ppg['times'][wd['peaklist']]

    times = closest_argmin(ecg_peak_times, ppg_peak_times)

    t = ppg['times'][np.array(wd['peaklist'])[times]]
    e = ecg['times'][ecg_peaks]

    pat = (t - e) / 10**6

    return pat, ecg_peak_times


if __name__ == "__main__":

    print("PTT")

    # for i in range(10):
    w = np.load("raw_data/data_5.npy", allow_pickle=True).item()

    ecg = [v for k, v in w.signals.items() if 'ECG' in k[0]][0]
    ecg_freq = [k[1] for k, v in w.signals.items() if 'ECG' in k[0]][0] / 10**9
    ppg = [v for k, v in w.signals.items() if 'PULS' in k[0]][0]
    ppg_freq = [k[1]
                for k, v in w.signals.items() if 'PULS' in k[0]][0] / 10**9
    abp = [v for k, v in w.signals.items() if 'ABP' in k[0]][0]

    ecg_peaks, ppg_peaks, match = get_ecg_signature(
        ecg, ecg_freq, ppg, ppg_freq)

    if True:
        fig, ax = plt.subplots(3, figsize=(25, 20))

        ax[0].plot(ecg['times'], ecg['values'])
        ax[0].plot(ecg['times'],
                   hp.filter_signal(ecg['values'], sample_rate=ecg_freq,
                                    cutoff=20, order=2, filtertype='lowpass'))
        ax[0].plot(ecg['times'][ecg_peaks],
                   ecg['values'][ecg_peaks], "x")

        ax[1].plot(ppg['times'], ppg['values'])
        ax[1].plot(ppg['times'][ppg_peaks],
                   ppg['values'][ppg_peaks], "x")

        ax[2].plot(ecg['times'][ecg_peaks], match)

        plt.show()
        # plt.savefig(f"plots/pat_{i}")

    # calculate_pat(ecg, ecg_freq/10**9, ppg, ppg_freq/10**9)
