from dataclasses import dataclass, field

import heartpy as hp
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
from biosppy.signals import ecg
from biosppy.signals.ppg import find_onsets_kavsaoglu2016
from scipy.linalg import norm
from scipy.spatial import distance

from plotting import plot_pat, plot_pat_hist


def ppg_peak_detect(signal_times, signal_values, freq_hz):
    """
    Get peaks from PPG

    :param signal: PPG signal
    :param freq: Frequency of signal

    :return: Indicies of peak onsets
    """

    # signal = hp.filter_signal(signal, sample_rate=freq,
    # cutoff=40, order=2, filtertype='lowpass')
    # working_data, measures = hp.process(signal_values, sample_rate=freq_hz)

    onsets = find_onsets_kavsaoglu2016(
        signal=signal_values,
        sampling_rate=freq_hz,
        init_bpm=90,
        min_delay=0.2,
        max_BPM=190,
    )[0]

    return signal_times[onsets]


def rpeak_detect_fast(signal_times, signal_values, freq_hz):
    """
    Detect R-peaks using NeuroKit2

    :param signal_values: Values of signal
    :param freq_hz: Frequency of signal

    :return: R-peak indices
    """

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

    return signal_times[corrected_peak_indices]


def closest_argmin(x, y):

    x = x[:, None]
    y = y[None, :]

    z = y - x

    z[np.where(z <= 0)] = np.max(z)

    return np.array(z.argmin(axis=-1)).astype(int)


def get_quality_index(signal, threshold=30):
    """
    Get quality of signal based on the interbeat intervals and change in HR

    :param signal: Signal to check
    :param threshold: Threshold for change in HR

    :return: Quality index
    """

    # Get IBI
    ibi = np.diff(signal)

    # Convert to HR
    hr = 60 / ibi

    # Get change in HR
    delta_hr = np.diff(hr)

    # Get quality index
    quality = np.absolute(delta_hr) < threshold

    return quality, hr


@dataclass
class MatchedPeak:
    ecg_peak: int = np.nan
    nearest_ppg_peak: int = np.nan
    distance: list[float] = field(default_factory=list)
    n_peaks: int = np.nan


def get_matching_peaks(ecg_peak_times, ppg_peak_times, wsize=20, ssize=6):
    """
    Align peaks from ECG and PPG signals

    :param ecg_peak_times: ECG peak times
    :param ppg_peak_times: PPG peak times
    :param wsize: Size of window to compare peaks
    :param ssize: Max number of peaks to search ahead

    :return: ECG peaks, PPG peaks, matching peaks
    """

    # Precalculate quality index for entire PPG signal
    ppg_quality, hr = get_quality_index(ppg_peak_times)

    matching_peaks = list()

    for i in range(ecg_peak_times.size - wsize - ssize):

        # Find nearest time in ppg after ecg peak to start search
        idx = np.where(ppg_peak_times > ecg_peak_times[i])[0][0]

        # Signal quality passed and within idx bounds for search
        if (
            idx + wsize + ssize < ppg_peak_times.size
            and ppg_quality[idx : idx + wsize + ssize].all()
        ):

            # Calculate distance for each PPG peak in search window
            euclidean = np.array(
                [
                    distance.euclidean(
                        # ECG interbeat signature
                        np.diff(ecg_peak_times[i : i + wsize]),
                        # PPG interbeat signature
                        np.diff(ppg_peak_times[idx + j : idx + j + wsize]),
                    )
                    for j in range(ssize)
                ]
            )

            m_peak = MatchedPeak(i, idx, euclidean, np.argmin(euclidean))
            matching_peaks.append(m_peak)

    return matching_peaks


def calclulate_pat(ecg, ecg_freq, ppg, ppg_freq, pat_range=0.200):
    """
    Calculate PAT

    :param ecg: ECG signal
    :param ecg_freq: Frequency of ECG signal
    :param ppg: PPG signal
    :param ppg_freq: Frequency of PPG signal
    :param pat_range: Range PAT can deviate from median within window (seconds)

    :return: Array of PAT
    """

    # Assert no nans in signal values (Causes peak detection to fail)
    # assert not np.isnan(ecg["values"]).any() == "ECG has NaNs"
    # assert not np.isnan(ppg["values"]).any() == "PPG has NaNs"

    ecg_peak_times = rpeak_detect_fast(ecg["times"], ecg["values"], ecg_freq)
    ppg_peak_times = ppg_peak_detect(ppg["times"], ppg["values"], ppg_freq)

    print(ecg_peak_times.size, ppg_peak_times.size)

    matching_peaks = get_matching_peaks(ecg_peak_times, ppg_peak_times)

    pats = np.array(
        [
            (
                m.ecg_peak,
                ppg_peak_times[m.nearest_ppg_peak + m.n_peaks]
                - ecg_peak_times[m.ecg_peak],
            )
            for m in matching_peaks
        ]
    )

    # TODO: Can attempt to correct the PAT values here by shifting the matching
    # peak forward/backwards

    # Remove PAT outliers
    baseline_pat = np.median(pats[:, 1])
    corrected_pats = pats[np.where(np.abs(pats[:, 1] - baseline_pat) < pat_range)]

    n_cleaned = pats.size - corrected_pats.size

    print(f"Median PAT: {baseline_pat}")
    print(f"Mean PAT: {np.mean(pats[:, 1])}")
    print(f"Removed {n_cleaned} outliers")
    print(f"Mean Corrected PAT: {np.mean(corrected_pats[:, 1])}")

    return corrected_pats, ecg_peak_times, ppg_peak_times, n_cleaned


def naive_calculate_pat(ecg, ecg_freq, ppg, ppg_freq):
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

    pats, ecg_peak_times, ppg_peak_times, n_cleaned = calclulate_pat(
        ecg_data, ecg_freq, ppg_data, ppg_freq
    )
    print("Finished calculating PAT...")

    # x = np.array([d[1] for d in dist])
    # y = np.array([d[0] for d in dist])
    # x_noise = np.random.normal(0, 0.1, x.size)

    # ax.plot(m_peaks[:-1] + x_noise, 60 / np.diff(ecg_peaks[: pats.size]), "x")
    # ax.plot(x + x_noise, y, "x")
    # ax.set_title("HR vs PAT")
    # ax.set_xlabel("Distance to matching PPG Peak (num peaks)")
    # ax.set_ylabel("Eculedian Distance")
    # plt.show()

    plot_pat(
        ecg_data,
        ecg_peak_times,
        ppg_data,
        ppg_peak_times,
        pats,
        show=True,
        save=False,
    )
