import warnings
from dataclasses import dataclass, field

import heartpy as hp
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd
from atriumdb import AtriumSDK
from biosppy.signals import abp, ecg
from biosppy.signals.ppg import find_onsets_kavsaoglu2016
from scipy.linalg import norm
from scipy.spatial import distance

from plotting.pat import plot_pat, plot_pat_hist


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


def get_quality_index(signal, min_hr=20, max_hr=300):
    """
    Get quality of signal based on the interbeat intervals and change in HR

    :param signal: Signal to check
    :param min_hr: Minimum HR
    :param max_hr: Maximum HR

    :return: Quality index
    """

    # Get IBI
    ibi = np.diff(signal)

    # Convert to HR
    hr = 60 / ibi

    # Get quality index
    hr_quality = (min_hr < hr) & (hr < max_hr)

    return hr_quality


@dataclass
class MatchedPeak:
    ecg_peak: int = np.nan
    nearest_ppg_peak: int = np.nan
    distance: list[float] = field(default_factory=list)
    n_peaks: int = np.nan
    ibi: float = np.nan
    confidence: float = np.nan
    possible_pats: list[float] = field(default_factory=list)


def beat_matching(ecg_peak_times, ppg_peak_times, wsize=20, ssize=6, max_search_time=2):
    """
    Align peaks from ECG and PPG signals

    :param ecg_peak_times: ECG peak times
    :param ppg_peak_times: PPG peak times
    :param wsize: Size of window to compare peaks
    :param ssize: Max number of peaks to search ahead

    :return: ECG peaks, PPG peaks, matching peaks
    """

    # Precalculate quality index for entire PPG signal
    # ecg_quality = get_quality_index(ecg_peak_times)
    # ppg_quality = get_quality_index(ppg_peak_times)

    matching_beats = list()

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
        if (idx + wsize + ssize) < ppg_peak_times.size:
            # and
            #     # ppg_quality[idx : idx + wsize + ssize].all()
            # ):

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

            # Return all possible PATs for each peak in search window
            possible_pats = np.array(
                [ppg_peak_times[idx + j] - ecg_peak_times[i] for j in range(ssize)]
            )

            ibi = np.diff(ecg_peak_times[i : i + wsize])
            confidence = np.sum(ibi) / euclidean[best_match]

            m_peak = MatchedPeak(
                i, idx, euclidean, best_match, ibi, confidence, possible_pats
            )
            matching_beats.append(m_peak)

    return matching_beats


def correct_pats(pats_df, matching_beats, pat_range=0.100):
    """
    Calculate Pulse Arrival Time

    :param matching_beats: Pandas df of PATs

    :return: corrected PATs
    """

    expected = np.median(pats_df["bm_pat"][pats_df["confidence"] > 0.7])
    print(f"Expected PAT: {expected}")

    pats_df["corrected_bm_pat"] = pats_df["bm_pat"]
    pats_df["valid_correction"] = np.ones(pats_df.shape[0]) * 2

    print(f"range: {expected - pat_range} - {expected + pat_range}")

    for row in pats_df[
        (pats_df["corrected_bm_pat"] < (expected - pat_range))
        | (pats_df["corrected_bm_pat"] > (expected + pat_range))
    ].iterrows():

        i = row[0]

        m = matching_beats[i]

        best = 0
        beats = 0
        for j, new_pat in enumerate(m.possible_pats):

            # Best PAT is selected as closest to expected value over search
            if abs(new_pat - expected) < abs(best - expected):
                best = new_pat
                beats = j

        if expected - pat_range < best < expected + pat_range:
            pats_df.loc[i, "corrected_bm_pat"] = best
            pats_df.loc[i, "valid_correction"] = 1
            pats_df.loc[i, "beats_skipped"] = beats

        # Couldn't find a good enough correction, remove point
        else:
            pats_df.loc[i, "valid_correction"] = 0
