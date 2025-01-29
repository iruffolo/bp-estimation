import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from beat_matching import beat_matching, correct_pats, peak_detect, rpeak_detect_fast
from kf_sawtooth import calw_sawtooth


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
    all_pats = pd.DataFrame(
        [m.possible_pats for m in matching_beats],
        columns=[f"{i + 1} beats" for i in range(ssize)],
    )
    all_pats["bm_pat"] = [m.possible_pats[m.n_peaks] for m in matching_beats]
    all_pats["confidence"] = [m.confidence for m in matching_beats]
    all_pats["times"] = [ecg_peak_times[m.ecg_peak] for m in matching_beats]
    all_pats["beats_skipped"] = [m.n_peaks for m in matching_beats]
    all_pats["age_days"] = all_pats["times"].apply(
        lambda x: (datetime.fromtimestamp(x) - dob).days
    )

    #### Correct Mismatched Beats ####
    correct_pats(all_pats, matching_beats, pat_range=0.100)
    df = all_pats[all_pats["valid_correction"] > 0]

    #### Sawtooth Correction ####
    fn = f"{w.patient_id}_{dev}_{i}"
    cdata, p1, p2 = calc_sawtooth(df["times"], df["corrected_bm_pat"], fn)

    st1 = pd.DataFrame(cdata["st1"])
    st2 = pd.DataFrame(cdata["st2"])

    st1["age_days"] = st1["times"].apply(
        lambda x: (datetime.fromtimestamp(x) - dob).days
    )
    st2["age_days"] = st2["times"].apply(
        lambda x: (datetime.fromtimestamp(x) - dob).days
    )

    for day in all_pats["age_days"][all_pats["age_days"] <= 7000].unique():
        naive = np.histogram(
            all_pats["1 beats"][all_pats["age_days"] == day],
            bins=bins,
            range=bin_range,
        )[0]
        hists["naive"][day] += naive

        bm = np.histogram(
            df["corrected_bm_pat"][df["age_days"] == day],
            bins=bins,
            range=bin_range,
        )[0]
        hists["bm"][day] += bm

        bm_st1 = np.histogram(
            st1["values"][st1["age_days"] == day],
            bins=bins,
            range=bin_range,
        )[0]
        hists["bm_st1"][day] += bm_st1

        bm_st1_st2 = np.histogram(
            st2["values"][st2["age_days"] == day],
            bins=bins,
            range=bin_range,
        )[0]
        hists["bm_st1_st2"][day] += bm_st1_st2

    log.log_raw_data(p1, f"st1_params")
    log.log_raw_data(p2, f"st2_params")
    log.log_status(WindowStatus.SUCCESS)
