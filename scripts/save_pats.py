from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pat import calculate_pat
from utils.logger import WindowStatus

matplotlib.use("Agg")


class SavePats:

    def __init__(self, device, log, bins=5000, bin_range=(0, 5)):

        self.log = log
        self.device = device

        self.bins = bins
        self.bin_range = bin_range

        self.cols = ["naive", "bm", "bm_st1", "bm_st1_st2"]
        self.data = {c: {} for c in self.cols}

        self.seen_patients = {}

        self.windows = 0

    def process_window(self, ecg, ecg_freq, ppg, ppg_freq, dob, pid):

        pats, st1, st2, ecg_peak_times, ppg_peak_times = calculate_pat(
            ecg, ecg_freq, ppg, ppg_freq, self.log
        )

        for d in [pats, st1, st2]:
            d["age_days"] = d["times"].apply(
                lambda x: (datetime.fromtimestamp(x) - dob).days
            )

        res = {
            "naive": pats[["naive", "age_days"]],
            "bm": pats[pats["valid_correction"] > 0],
            "bm_st1": st1,
            "bm_st1_st2": st2,
        }

        for age in pats["age_days"].unique():
            for name in self.cols:
                h = np.histogram(
                    res[name][res[name]["age_days"] == age],
                    bins=self.bins,
                    range=self.bin_range,
                )[0]

                if age in self.data[name]:
                    self.data[name][age]["h"] += h
                    self.data[name][age]["num_windows"] += 1

                else:
                    self.data[name][age] = {
                        "h": h,
                        "num_patients": 0,
                        "num_windows": 1,
                        "num_measurements": 0,
                        "seen_patients": [],
                    }

                if pid not in self.data[name][age]["seen_patients"]:
                    self.data[name][age]["num_patients"] += 1
                    self.data[name][age]["seen_patients"].append(pid)

        self.log.log_raw_data(st1, f"st1_params")
        self.log.log_raw_data(st2, f"st2_params")

        m = np.median(res["bm"]["bm_pat"])
        if m < 1.2:
            self.debug_plot(
                ecg,
                ppg,
                pats,
                ecg_peak_times,
                ppg_peak_times,
                path="../data/debug_plots/bad/",
            )
        else:
            self.debug_plot(ecg, ppg, pats, ecg_peak_times, ppg_peak_times)

        self.windows += 1

    def debug_plot(
        self,
        ecg,
        ppg,
        pats,
        ecg_peak_times,
        ppg_peak_times,
        path="../data/debug_plots/valid/",
    ):

        fig, ax = plt.subplots(6, 1, figsize=(15, 10))

        bm = pats[pats["valid_correction"] > 0]

        ax[0].plot(ecg["times"], ecg["values"])
        ax[0].set_title("ECG")
        ax[1].plot(ppg["times"], ppg["values"])
        ax[1].set_title("PPG")

        ax[2].scatter(ecg_peak_times[:-1], np.diff(ecg_peak_times), marker="x")
        ax[2].set_ylim((0, 2))
        ax[2].set_title("ECG IBIs")
        ax[3].scatter(ppg_peak_times[:-1], np.diff(ppg_peak_times), marker="x")
        ax[3].set_ylim((0, 2))
        ax[3].set_title("PPG IBI")

        ax[4].scatter(bm["times"], bm["bm_pat"])
        ax[4].set_ylim((0, 2))
        ax[4].set_title("Beat Matching")
        ax[5].scatter(bm["times"], bm["corrected_bm_pat"], marker="x")
        ax[5].set_ylim((0, 2))
        ax[5].set_title("Beat Matching w/ Correction")

        plt.tight_layout()
        plt.savefig(path + f"{self.device}_{self.windows}.png")
