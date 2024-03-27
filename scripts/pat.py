import heartpy as hp
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import seaborn as sns
from biosppy.signals import ecg
from biosppy.signals.ppg import find_onsets_kavsaoglu2016
# https://github.com/LaussenLabs/consensus_peaks
# from consensus_peaks import consensus_detect
from scipy.spatial import distance


def ppg_peak_detect(signal_times, signal_values, freq_hz):
    """
    Get peaks from PPG

    :param signal: PPG signal
    :param freq: Frequency of signal

    :return: Position of peak onsets
    """

    # signal = hp.filter_signal(signal, sample_rate=freq,
    # cutoff=40, order=2, filtertype='lowpass')
    # working_data, measures = hp.process(signal_values, sample_rate=freq_hz)

    # print(working_data.keys())
    # print(measures.keys())

    onsets = find_onsets_kavsaoglu2016(
        signal=signal_values,
        sampling_rate=freq_hz,
        init_bpm=90,
        min_delay=0.2,
        max_BPM=190,
    )[0]

    return signal_times[onsets], onsets


def rpeak_detect_fast(signal_times, signal_values, freq_hz):
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
        pass
        # print(f"Error: {e}")

    peak_indices = info["ECG_R_Peaks"]

    # Correct Peaks
    (corrected_peak_indices,) = ecg.correct_rpeaks(
        signal=signal_values, rpeaks=peak_indices, sampling_rate=freq_hz
    )

    return signal_times[corrected_peak_indices], corrected_peak_indices


def closest_argmin(x, y):

    x = x[:, None]
    y = y[None, :]

    z = y - x

    z[np.where(z <= 0)] = np.max(z)

    return np.array(z.argmin(axis=-1)).astype(int)


def get_quality_index(signal, threshold=20):
    """
    Get quality of signal based on the interbeat intervals and change in HR

    :param signal: Signal to check

    :return: Quality index
    """

    # Get IBI
    ibi = np.diff(signal)
    print(ibi)

    # Convert to HR
    hr = 60 / ibi

    # Get change in HR
    delta_hr = np.diff(hr)

    # Get quality index
    quality = np.absolute(delta_hr) < threshold

    return quality, hr


def align_peaks(ecg, ecg_freq, ppg, ppg_freq, wsize=15, ssize=6):
    """
    Align peaks from ECG and PPG signals

    :param ecg: ECG signal
    :param ecg_freq: Frequency of ECG signal (Hz)
    :param ppg: PPG signal
    :param ppg_freq: Frequency of PPG signal (Hz)
    :param wsize: Size of window to compare peaks
    :param ssize: Max number of peaks to search ahead

    :return: ECG peaks, PPG peaks, matching peaks
    """

    ecg_peak_times, idx_ecg = rpeak_detect_fast(ecg["times"], ecg["values"], ecg_freq)
    ppg_peak_times, idx_ppg = ppg_peak_detect(ppg["times"], ppg["values"], ppg_freq)

    ecg_quality, _ = get_quality_index(ecg_peak_times)
    ppg_quality, hr = get_quality_index(ppg_peak_times)

    matching_peaks = list()
    pats = list()
    for i in range(ecg_peak_times.size - wsize - ssize):

        if ecg_quality[i]:
            try:
                # Find nearest time in ppg to ecg peak to start search
                idx = np.absolute(ppg_peak_times - ecg_peak_times[i]).argmin()

                # Compare window of peaks
                dist = np.array(
                    [
                        distance.euclidean(
                            np.diff(ecg_peak_times[i : i + wsize]),
                            np.diff(ppg_peak_times[idx + j : idx + j + wsize]),
                        )
                        for j in range(ssize)
                    ]
                )

                min = np.argmin(dist)

                # Quality passed, calcualte PAT
                if ppg_quality[idx : idx + min + wsize + ssize].all():
                    pat = ppg_peak_times[idx + min] - ecg_peak_times[i]

                else:
                    pat = np.nan

                matching_peaks.append(min)
                pats.append(pat)

            except Exception as e:
                print(f"Error in aligntment: {e}")
                matching_peaks.append(np.nan)
                pats.append(np.nan)
        else:
            matching_peaks.append(np.nan)
            pats.append(np.nan)

    matching_peaks = np.array(matching_peaks)
    pats = np.array(pats)

    return (ecg_peak_times, ppg_peak_times, idx_ecg, idx_ppg, matching_peaks, pats, hr)


def calculate_pat(ecg, ecg_freq, ppg, ppg_freq):
    """
    Naive calculation of PAT
    """

    ecg_peak_times, idx_ecg = rpeak_detect_fast(ecg["times"], ecg["values"], ecg_freq)
    ppg_peak_times, idx_ppg = ppg_peak_detect(ppg["times"], ppg["values"], ppg_freq)

    times = closest_argmin(ecg_peak_times, ppg_peak_times)

    t = ppg["times"][np.array(wd["peaklist"])[times]]
    e = ecg["times"][ecg_peaks]

    pat = (t - e) / 10**6

    return pat, ecg_peak_times


def rpeak_dist(ecg, freq):

    peaks = rpeak_detect_fast(ecg["times"], ecg["values"], freq) / 10**9
    # print(peaks)

    diff = np.diff(peaks)

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


if __name__ == "__main__":

    print("Calculating PAT...")

    w = np.load("raw_data/data_0_hourly.npy", allow_pickle=True).item()

    ecg_data, ecg_freq = [
        (v, k[1] / 10**9) for k, v in w.signals.items() if "ECG" in k[0]
    ][0]
    ppg_data, ppg_freq = [
        (v, k[1] / 10**9) for k, v in w.signals.items() if "PULS" in k[0]
    ][0]

    ecg_data["times"] = ecg_data["times"] / 10**9
    ppg_data["times"] = ppg_data["times"] / 10**9

    ecg_peaks, ppg_peaks, idx_ecg, idx_ppg, m_peaks, pats, hr = align_peaks(
        ecg_data, ecg_freq, ppg_data, ppg_freq
    )

    clean_pats = pats[(pats > 0.5) & (pats < 2)]

    plot_pat(
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
        save=False,
    )

    # plot_pat_hist(clean_pats, show=True, save=False)
