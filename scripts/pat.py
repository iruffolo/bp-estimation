import warnings
from dataclasses import dataclass, field

import heartpy as hp
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
from atriumdb import AtriumSDK
from biosppy.signals import abp, ecg
from biosppy.signals.ppg import find_onsets_kavsaoglu2016
from plotting.pat import plot_pat, plot_pat_hist
from scipy.linalg import norm
from scipy.spatial import distance


def peak_detect(signal_times, signal_values, freq_hz):
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

        peak_indices = info["ECG_R_Peaks"]

        # Correct Peaks
        (corrected_peak_indices,) = ecg.correct_rpeaks(
            signal=signal_values, rpeaks=peak_indices, sampling_rate=freq_hz
        )

        return signal_times[corrected_peak_indices]

    except Exception as e:
        print(f"Error: {e}")
        return np.array([])


def closest_argmin(x, y):

    x = x[:, None]
    y = y[None, :]

    z = y - x

    z[np.where(z <= 0)] = np.max(z)

    return np.array(z.argmin(axis=-1)).astype(int)


def get_quality_index(signal, min_hr=20, max_hr=300, diff_threshold=50):
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
    dt_quality = np.absolute(delta_hr) < diff_threshold
    hr_quality = (min_hr < hr) & (hr < max_hr)

    quality = dt_quality & hr_quality[:-1]
    return quality


@dataclass
class MatchedPeak:
    ecg_peak: int = np.nan
    nearest_ppg_peak: int = np.nan
    distance: list[float] = field(default_factory=list)
    n_peaks: int = np.nan


def get_matching_peaks(
    ecg_peak_times, ppg_peak_times, wsize=20, ssize=6, max_search_time=1
):
    """
    Align peaks from ECG and PPG signals

    :param ecg_peak_times: ECG peak times
    :param ppg_peak_times: PPG peak times
    :param wsize: Size of window to compare peaks
    :param ssize: Max number of peaks to search ahead

    :return: ECG peaks, PPG peaks, matching peaks
    """

    # Precalculate quality index for entire PPG signal
    ppg_quality = get_quality_index(ppg_peak_times)

    matching_peaks = list()

    for i in range(ecg_peak_times.size - wsize):

        # Find nearest time in ppg after ecg peak to start search, within limit
        try:
            idx = np.where(
                (ppg_peak_times > ecg_peak_times[i])
                & (ppg_peak_times - ecg_peak_times[i] < max_search_time)
            )[0][0]
        except:
            # No corresponding peak in PPG signal (likely at end of window)
            continue

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
            best_match = np.argmin(euclidean)

            m_peak = MatchedPeak(i, idx, euclidean, best_match)
            matching_peaks.append(m_peak)

    return matching_peaks


def calclulate_pat(ecg, ecg_freq, ppg, ppg_freq, pat_range=0.250):
    """
    Calculate Pulse Arrival Time

    :param ecg: ECG signal
    :param ecg_freq: Frequency of ECG signal
    :param ppg: PPG signal
    :param ppg_freq: Frequency of PPG signal
    :param pat_range: Range PAT can deviate from median within window (seconds)
    :param ssize: Number of peaks to search ahead

    :return: Array of PAT
    """

    # Find peaks in ECG and PPG signals
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ecg_peak_times = rpeak_detect_fast(ecg["times"], ecg["values"], ecg_freq)
        ppg_peak_times = peak_detect(ppg["times"], ppg["values"], ppg_freq)

    # assert ecg_peak_times.size > 500, "Not enough ECG peaks found"
    # assert ppg_peak_times.size > 500, "Not enough PPG peaks found"

    # Match ppg peaks to ecg peaks
    ssize = 6
    matching_peaks = get_matching_peaks(ecg_peak_times, ppg_peak_times, ssize=ssize)

    # Calculate PAT for each matched peak
    pats = {
        "times": np.zeros([len(matching_peaks)]),
        "values": np.zeros([len(matching_peaks)]),
    }
    naive_pats = {
        "times": np.zeros([len(matching_peaks)]),
        "values": np.zeros([len(matching_peaks)]),
    }

    for i, m in enumerate(matching_peaks):
        pat = (
            ppg_peak_times[m.nearest_ppg_peak + m.n_peaks] - ecg_peak_times[m.ecg_peak]
        )

        pats["times"][i] = ecg_peak_times[m.ecg_peak]
        pats["values"][i] = pat
        naive_pats["times"][i] = ecg_peak_times[m.ecg_peak]
        naive_pats["values"][i] = (
            ppg_peak_times[m.nearest_ppg_peak] - ecg_peak_times[m.ecg_peak]
        )

    # Use median as expected value to correct outliers (i.e. mismatched beats)
    expected = np.median(pats["values"])

    if expected < 0.5 or expected > 2:
        print(pats)
        print(naive_pats)

        print(f"pats: {len(pats)}")
        print(f"naive_pats: {len(naive_pats)}")
        print(f"matching_peaks: {len(matching_peaks)}")
        print(f"ecg_peak_times: {len(ecg_peak_times)}")
        print(f"ppg_peak_times: {len(ppg_peak_times)}")
        exit()

    num_corrected = 0
    tobedeleted = list()

    for i, pat in enumerate(pats["values"]):

        # PAT outside expected range, correct it
        if not (expected - pat_range < pat < expected + pat_range):

            m = matching_peaks[i]

            best = 0
            for s in range(ssize):
                new_pat = (
                    ppg_peak_times[m.nearest_ppg_peak + s] - ecg_peak_times[m.ecg_peak]
                )

                # Best PAT is selected as closest to expected value over search
                if abs(new_pat - expected) < abs(best - expected):
                    best = new_pat

            if expected - pat_range < best < expected + pat_range:
                # print(f"Good correction {pat} -> {best}")
                pats["values"][i] = best
                num_corrected += 1

            # Couldn't find a good enough correction, remove point
            else:
                # print(f"Cant find correction, destroying {pat}")
                tobedeleted.append(i)

    pats["times"] = np.delete(pats["times"], tobedeleted)
    pats["values"] = np.delete(pats["values"], tobedeleted)
    # naive_pats["times"] = np.delete(naive_pats["times"], tobedeleted)
    # naive_pats["values"] = np.delete(naive_pats["values"], tobedeleted)

    return pats, naive_pats, num_corrected, ecg_peak_times, ppg_peak_times


def calc_pat_abp(ecg, ecg_freq, abp, abp_freq):
    """ """

    ecg_peak_times = rpeak_detect_fast(ecg["times"], ecg["values"], ecg_freq)
    abp_peak_times = peak_detect(abp["times"], abp["values"], abp_freq)

    pats = []

    for p in ecg_peak_times:
        try:
            idx = np.where(abp_peak_times > p)[0][0]

            pats.append((idx, abp_peak_times[idx] - p))

        except Exception as e:
            print(f"Error: {e}")

    return np.array(pats), ecg_peak_times, abp_peak_times


def naive_calculate_pat(ecg, ecg_freq, ppg, ppg_freq):
    """
    Naive calculation of PAT
    """

    ecg_peak_times = rpeak_detect_fast(ecg["times"], ecg["values"], ecg_freq)
    ppg_peak_times = peak_detect(ppg["times"], ppg["values"], ppg_freq)

    times = closest_argmin(ecg_peak_times, ppg_peak_times)

    t = ppg["times"][np.array(wd["peaklist"])[times]]
    e = ecg["times"][ecg_peaks]

    pat = (t - e) / 10**6

    return pat, ecg_peak_times


if __name__ == "__main__":

    print("Calculating PAT...")

    # w = np.load("raw_data/data_0_hourly.npy", allow_pickle=True).item()
    local_dataset = "/mnt/datasets/ians_data_2024_06_12"
    sdk = AtriumSDK(dataset_location=local_dataset)

    ecg_data, ecg_freq = [
        (v, k[1] / 10**9) for k, v in w.signals.items() if "ECG" in k[0]
    ][0]
    ppg_data, ppg_freq = [
        (v, k[1] / 10**9) for k, v in w.signals.items() if "PULS" in k[0]
    ][0]
    abp_data, abp_freq = [
        (v, k[1] / 10**9) for k, v in w.signals.items() if "ABP" in k[0]
    ][0]

    ecg_data["times"] = ecg_data["times"] / 10**9
    ppg_data["times"] = ppg_data["times"] / 10**9
    abp_data["times"] = abp_data["times"] / 10**9

    print(f"Freqs: {ecg_freq}, {ppg_freq}, {abp_freq}")

    pats, ecg_peak_times, ppg_peak_times, n_cleaned = calclulate_pat(
        ecg_data, ecg_freq, ppg_data, ppg_freq
    )
    # abp_pats, _, abp_peak_times, abp_n_cleaned = calclulate_pat(
    # ecg_data, ecg_freq, abp_data, abp_freq
    # )

    abp_pats, _, abp_peak_times, _ = calc_pat_abp(
        ecg_data, ecg_freq, abp_data, abp_freq
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
        abp_data,
        abp_peak_times,
        pats,
        abp_pats,
        show=True,
        save=False,
    )
