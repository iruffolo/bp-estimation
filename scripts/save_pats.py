import numpy as np
import pandas as pd
from pat import calculate_pat


class SavePats:

    def __init__(self, log, bins=5000, bin_range=(0, 5), days=7000):

        self.log = log

        self.max_days = 7000

        # Daily up to 18 years
        age_bins = np.linspace(0, self.max_days, self.max_days + 1)

        cols = ["naive", "bm", "bm_st1", "bm_st1_st2"]

        self.data = {
            n: {
                age: {
                    h: np.histogram([], bins=bins, range=bin_range)[0],
                    "num_patients": 0,
                    "num_measurements": 0,
                }
                for age in age_bins
            }
        }

    def process_window(self, ecg, ecg_freq, ppg, ppg_freq):

        pat, st1, st2 = calculate_pat(ecg, ecg_freq, ppg, ppg_freq, self.log)

        for day in pats["age_days"][pats["age_days"] <= self.max_days].unique():

            naive = np.histogram(
                pats["1 beats"][pats["age_days"] == day],
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
