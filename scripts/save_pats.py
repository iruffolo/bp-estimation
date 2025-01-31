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

        fig, ax = plt.subplots(3, 1, figsize=(25, 20))

        bm = pats[pats["valid_correction"] > 0]
        bm = pats

        # ax[0].plot(ecg["times"], ecg["values"])
        # ax[0].set_title("ECG")
        # ax[1].plot(ppg["times"], ppg["values"])
        # ax[1].set_title("PPG")

        ecg_ibi = np.diff(ecg_peak_times) * 1000
        ecg_ibi_diff = np.diff(np.diff(ecg_peak_times)) * 1000
        ppg_ibi = np.diff(ppg_peak_times) * 1000
        ppg_ibi_diff = np.diff(np.diff(ppg_peak_times)) * 1000

        ax[0].scatter(ecg_peak_times[:-1], ecg_ibi, marker=".", alpha=0.8, label="ECG")
        ax[0].set_ylim((0, 1000))
        ax[0].set_title("ECG and PPG IBIs")

        ax[0].scatter(ppg_peak_times[:-1], ppg_ibi, marker=".", alpha=0.8, label="PPG")
        # ax[1].set_ylim((0, 1000))
        # ax[1].set_title("PPG IBI")
        ax[0].legend(loc="upper right")

        ax[1].scatter(
            ecg_peak_times[:-2], ecg_ibi_diff, marker=".", s=20, alpha=0.5, label="ECG"
        )
        ax[1].scatter(
            ppg_peak_times[:-2], ppg_ibi_diff, marker=".", s=20, alpha=0.5, label="PPG"
        )
        ax[1].set_ylim((-50, 50))
        ax[1].set_title("Delta PPG & ECG IBI")
        ax[1].set_yticks(np.linspace(-50, 50, 11))
        ax[1].legend(loc="upper right")

        ax[2].scatter(bm["times"], bm["bm_pat"], marker=".", s=5)
        ax[2].set_title("Beat Matching")
        ax[2].scatter(bm["times"], bm["corrected_bm_pat"], marker=".", s=5, alpha=0.2)
        ax[2].set_title("Beat Matching w/ Correction")
        ax[2].set_ylim((0, 4))

        plt.grid(visible=True, which="both")

        plt.tight_layout()
        # plt.show(g
        # plt.savefig(path + f"{self.device}_{self.windows}.png")
        plt.close()

        ecgdf = pd.DataFrame(ecg_ibi_diff)
        ecgdf.to_csv(path + f"{self.device}_{self.windows}_ecg.csv", index=False)
        ppgdf = pd.DataFrame(ppg_ibi_diff)
        ppgdf.to_csv(path + f"{self.device}_{self.windows}_ppg.csv", index=False)

        fig, ax = plt.subplots(1, 2, figsize=(25, 20))
        ax[0].hist(ecg_ibi_diff, bins=40, range=(-20, 20))
        ax[0].set_xlabel("ECG IBI (ms)")
        ax[1].hist(ppg_ibi_diff, bins=200, range=(-100, 100))
        ax[1].set_xlabel("PPG IBI (ms)")
        plt.savefig(path + f"{self.device}_{self.windows}_hist.png")
        plt.close()
