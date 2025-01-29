import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from beat_matching import beat_matching, correct_pats
from kf_sawtooth import calc_sawtooth
from peak_extract import ppg_peak_detect, rpeak_detect_fast


def calculate_pat(ecg, ecg_freq, ppg, ppg_freq, log):
    """
    Given ECG and PPG raw data:
        1) Extract peaks from both waveforms
        2) Run beat matching algorithm
        3) Correct mismatched beats
        4) Apply sawtooth correction
    """

    #### Extract Peaks ####
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ecg_peak_times = rpeak_detect_fast(ecg["times"], ecg["values"], ecg_freq)
        ppg_peak_times = peak_detect(ppg["times"], ppg["values"], ppg_freq)

    assert ecg_peak_times.size > 500, "Not enough ECG peaks found"
    assert ppg_peak_times.size > 500, "Not enough PPG peaks found"

    #### Beat Matching ####
    ssize = 6
    matching_beats = beat_matching(ecg_peak_times, ppg_peak_times, ssize=ssize)
    assert len(matching_beats) > 0, "BM failed to find any matching beats"

    log.log_status(WindowStatus.TOTAL_BEATS, len(ecg_peak_times))
    log.log_status(
        WindowStatus.TOTAL_BEATS_DROPPED,
        len(ecg_peak_times) - len(matching_beats),
    )

    # Create a df for all possible PAT values
    pats = pd.DataFrame(
        [m.possible_pats for m in matching_beats],
        columns=[f"{i + 1}_beat" for i in range(ssize)],
    )

    pats["naive"] = [m.possible_pats[0] for m in matching_beats]
    pats["bm_pat"] = [m.possible_pats[m.n_peaks] for m in matching_beats]
    pats["confidence"] = [m.confidence for m in matching_beats]
    pats["times"] = [ecg_peak_times[m.ecg_peak] for m in matching_beats]
    pats["beats_skipped"] = [m.n_peaks for m in matching_beats]
    pats["age_days"] = pats["times"].apply(
        lambda x: (datetime.fromtimestamp(x) - dob).days
    )

    #### Correct Mismatched Beats ####
    correct_pats(pats, matching_beats, pat_range=0.100)
    df = pats[pats["valid_correction"] > 0]

    #### Sawtooth Correction ####
    st1, st2, p1, p2 = calc_sawtooth(df["times"], df["corrected_bm_pat"])

    st1["age_days"] = st1["times"].apply(
        lambda x: (datetime.fromtimestamp(x) - dob).days
    )
    st2["age_days"] = st2["times"].apply(
        lambda x: (datetime.fromtimestamp(x) - dob).days
    )

    return pats, st1, st2
