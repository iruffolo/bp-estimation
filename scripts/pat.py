import matplotlib.pyplot as plt
import numpy as np
import heartpy as hp
from scipy.spatial import distance
# https://github.com/LaussenLabs/consensus_peaks
from consensus_peaks import consensus_detect
import neurokit2 as nk
from biosppy.signals import ecg
from biosppy.signals.ppg import find_onsets_kavsaoglu2016 as ppg_onsets


def _get_ppg_peaks(signal, freq):
    """
    Get peaks from PPG
    """

    # signal = hp.filter_signal(signal, sample_rate=freq,
                              # cutoff=40, order=2, filtertype='lowpass')
    # working_data, measures = hp.process(signal, freq)

    onsets, _ = find_onsets_kavsaoglu2016(signal=signal, sampling_rate=freq, 
                              init_bpm=90, min_delay=0.6, max_BPM=250)

    return onsets


def neurokit_rpeak_detect_fast(signal_times, signal_values, freq_hz):
    """
    Detect R-peaks using NeuroKit2

    :param signal_times: Times of signal
    :param signal_values: Values of signal
    :param freq_hz: Frequency of signal
    """

    assert signal_times.size == signal_values.size
    try:
        clean_signal = nk.ecg_clean(signal_values, sampling_rate=int(freq_hz))
        signals, info = nk.ecg_peaks(clean_signal, sampling_rate=int(freq_hz))
    except Exception as e:
        print(f"Error: {e}")

    peak_indices = info["ECG_R_Peaks"]
    # Correct Peaks
    (corrected_peak_indices,) = ecg.correct_rpeaks(
        signal=signal_values, rpeaks=peak_indices, sampling_rate=freq_hz)

    return signal_times[corrected_peak_indices]


def _get_ecg_peaks(signal, freq):
    """
    Get peaks from ECG
    """

    # signal = hp.filter_signal(signal, sample_rate=freq,
                              # cutoff=40, order=2, filtertype='lowpass')

    c_peaks = consensus_detect(signal, freq)

    return c_peaks


def closest_argmin(x, y):
    x = x[:, None]
    y = y[None, :]

    z = y-x

    z[np.where(z <= 0)] = np.max(z)

    return np.array(z.argmin(axis=-1)).astype(int)


def get_ecg_signature(ecg, ecg_freq, ppg, ppg_freq, size=10):

    ecg_peaks = neurokit_rpeak_detect_fast(ecg['times'], ecg['values'], freq) / 10**9
    ecg_peaks = _get_ecg_peaks(ecg['values'][0:50000], ecg_freq)
    ecg_times = ecg['times'][ecg_peaks]
    ecg_diffs = [np.roll(ecg_times, -i) - ecg_times for i in range(size)]
    # Stack so each row represents a peak and its diffs
    ecg_diffs = np.stack(ecg_diffs, axis=1)  # [0]

    # Only keep valid diffs where peaks didnt wrap around
    # ecg_diffs = ecg_diffs[ecg_diffs.min(axis=1) >= 0, :]

    ppg_peaks = _get_ppg_peaks(ppg['values'][0:125000], ppg_freq)
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


def plot_all(abp, ecg, ppg, save=False, show=True):

    fig, ax = plt.subplots(4, figsize=(25, 15))

    ax[0].plot(abp['times'], abp['values'])
    ax[0].set_title("ABP")

    ax[1].plot(ecg['times'], ecg['values'])
    ax[1].set_title("ECG")

    ax[2].plot(ppg['times'], ppg['values'])
    ax[2].set_title("PPG")

    # ax[3].plot(ecg_peak_times, pat)
    # ax[3].set_title("PAT")

    if save:
        plt.savefig("pat")
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



def rpeak_dist(ecg, freq):

    peaks = neurokit_rpeak_detect_fast(ecg['times'], ecg['values'], freq) / 10**9
    # print(peaks)

    diff = np.diff(peaks)

    plt.hist(diff, bins=100)
    plt.title("R-Peak IBI Distribution")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency")
    plt.show()
    plt.savefig("plots/rpeak_ibi_dist")

    return diff


if __name__ == "__main__":

    print("Calculating PAT...")

    w = np.load("raw_data/data_0_hourly.npy", allow_pickle=True).item()

    ecg_data, ecg_freq = [(v, k[1]/10**9) for
                          k, v in w.signals.items() if 'ECG' in k[0]][0]
    ppg_data, ppg_freq = [(v, k[1]/10**9) for
                          k, v in w.signals.items() if 'PULS' in k[0]][0]

    rp = rpeak_dist(ecg_data, ecg_freq)
    exit()

    ecg_peaks, ppg_peaks, match = get_ecg_signature(
        ecg, ecg_freq, ppg_data, ppg_freq)

    def shift_times(times, scale=10**9):
        times = times - times[0]
        return times / scale

    if True:
        fig, ax = plt.subplots(3, figsize=(25, 20))

        ax[0].plot(shift_times(ecg_data['times']), ecg_data['values'])
        ax[0].plot(shift_times(ecg_data['times'][ecg_peaks]),
                   ecg_data['values'][ecg_peaks], "x")

        ax[1].plot(shift_times(ppg_data['times']), ppg_data['values'])
        ax[1].plot(shift_times(ppg_data['times'][ppg_peaks]),
                   ppg_data['values'][ppg_peaks], "x")

        ax[2].plot(shift_times(ecg_data['times'][ecg_peaks]), match)

        plt.show()
        # plt.savefig(f"plots/pat_{i}")

    # calculate_pat(ecg, ecg_freq/10**9, ppg, ppg_freq/10**9)
