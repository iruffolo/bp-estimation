import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from atriumdb import AtriumSDK
from beat_matching import beat_matching, correct_pats, peak_detect, rpeak_detect_fast
from bm_gui import ApplicationWindow
from matplotlib.backends.qt_compat import QtWidgets
from new_st import create_sawtooth, fit_sawtooth
from utils.atriumdb_helpers import (
    get_ppg_ecg_data,
    get_ppg_ecg_intervals_device,
    print_all_measures,
)


class Pats:
    def __init__(self, datapath, verbose=False):
        self.datapath = datapath

        self.verbose = verbose

        # Window size for initial data slice
        self.window_size_sec = 60 * 60
        self.ssize = 6

        self.ecg_freq = 500
        self.ppg_freq = 125

        # Load file names from path, extract patient ids and device ids
        self.process_files()
        self.print_stats()

        self.devices = self.p_df["dev"].unique()

    def print_stats(self):
        """
        Prints some stats about the dataset
        """
        print(f"Total patients: {self.p_df['pid'].nunique()}")
        print(f"Total devices: {self.p_df['dev'].nunique()}")

    def process_files(self):
        """
        Process all files in the dataset
        """

        patients = {
            "pid": [],
            "dev": [],
            "month": [],
            "year": [],
            "files": [],
        }

        for file in os.listdir(self.datapath):
            if not file.endswith("log.csv"):
                _, dev, pid, month, year, ftype = file.split("_")

                patients["pid"].append(int(pid))
                patients["dev"].append(int(dev))
                patients["month"].append(int(month))
                patients["year"].append(int(year))
                patients["files"].append(file)

        self.p_df = pd.DataFrame(patients)

    def update_patient_data(self):
        """
        Get new patient data from database, calculate inital beat matching and
        PATs.
        """

        rows = self.p_df[
            (self.p_df["pid"] == self.pid) & (self.p_df["dev"] == self.dev)
        ]

        self.year = rows["year"].values[0]
        self.month = rows["month"].values[0]

        print("Getting raw data")
        ecg_file = rows["files"][self.p_df["files"].str.contains("ecg")].values[0]
        ppg_file = rows["files"][self.p_df["files"].str.contains("ppg")].values[0]

        self.print("Extracting Peaks")
        self.ecg_beats = pd.read_csv(self.datapath + ecg_file).values.flatten()
        self.ppg_beats = pd.read_csv(self.datapath + ppg_file).values.flatten()

        self.app.set_title(f"Patient {self.pid} Date: {self.month}/{self.year}")

        # Remove offset from times
        self.ppg_beats = self.ppg_beats - self.ppg_beats[0]
        self.ecg_beats = self.ecg_beats - self.ecg_beats[0]

        idx = np.where((self.ecg_beats <= self.window_size_sec))[0]
        assert len(idx) > 0, "No peaks in window"
        self.ecg_peak_times = self.ecg_beats[idx]

        idx = np.where((self.ppg_beats <= self.window_size_sec))[0]
        assert len(idx) > 0, "No peaks in window"
        self.ppg_peak_times = self.ppg_beats[idx]

        self.app.plot_raw_data(self.ppg_peak_times, self.ecg_peak_times)

        # Calculate beat matching and PATs
        print("Starting BM")
        self.beat_matching()
        print("BM Done")

        # Plot all the results
        self.app.plot_beat_match_data(self.all_pats)
        self.app.plot_pat_data(self.pats, self.c_pats_df, self.cmap)

        # Create smaller window slice for sawtooth fitting
        self.start_time = 0
        self.update_window()

        print("Fitting Sawtooths")
        self.sawtooth_one()
        self.sawtooth_two()

        self.update_sawtooth_plots()

    def update_sawtooth_plots(self):
        app.plot_sawtooth_one(self.x, self.y, self.x_ls, self.y_st1, self.fitp1)
        app.plot_sawtooth_two(self.x, self.fixed_st1, self.x_ls, self.y_st2, self.fitp2)
        app.plot_corrected(self.x, self.corrected_window)

        self.app.update_button_text(self.fitp1, self.fitp2, self.window_s / 60)

    def update_window(self):
        """
        Update the window size and start time for the data slice
        """

        x = self.c_pats_df["times"].reset_index(drop=True)
        y = self.c_pats_df["values"].reset_index(drop=True)

        # Shift the pats to the left based on window size
        end_time = self.start_time + self.window_s

        print(self.start_time, end_time)
        idx = np.where((x >= self.start_time) & (x <= end_time))[0]
        if not idx.size:
            print("Window out of range, do nothing")
            return

        self.x = x.iloc[idx]
        self.y = y.iloc[idx]

        # Create shifted and scaled x values for sawtooth fitting
        self.x_shift = self.x.values - self.x.iloc[0]

        self.x_ls = np.linspace(min(self.x_shift), max(self.x_shift), num=500)

        # Shift x values back to original for plotting
        self.x_st_plot = self.x_ls + self.x.iloc[0]

    def beat_matching(self):
        """ """

        self.matching_beats = beat_matching(
            self.ecg_peak_times, self.ppg_peak_times, ssize=self.ssize
        )

        # Create a df for plotting all possible beat matching PATs
        self.all_pats = pd.DataFrame(
            [m.possible_pats for m in self.matching_beats],
            columns=[f"{i + 1} beats" for i in range(self.ssize)],
        )
        self.all_pats["times"] = [
            self.ecg_peak_times[m.ecg_peak] for m in self.matching_beats
        ]
        self.all_pats.set_index("times", inplace=True)

        # Select the best match in the beatmatching for actual PAT values
        self.pats = {
            "times": [self.ecg_peak_times[m.ecg_peak] for m in self.matching_beats],
            "values": [m.possible_pats[m.n_peaks] for m in self.matching_beats],
            "confidence": [m.confidence for m in self.matching_beats],
        }

        # Correct the PATs
        self.c_pats_df, _ = correct_pats(
            self.pats, self.matching_beats, pat_range=0.100
        )

        cmap = [m.confidence for m in self.matching_beats]
        self.cmap = (np.array(cmap) - np.min(cmap)) / (np.max(cmap) - np.min(cmap))

    def sawtooth_one(self, fit=True):

        if fit:
            self.st1, self.fitp1 = fit_sawtooth(self.x_shift, self.y)
        else:
            self.st1 = create_sawtooth(self.x_shift, *self.fitp1)

        self.fixed_st1 = (self.y - self.st1) + self.fitp1[2]

        # Sawtooth y values for plotting
        self.y_st1 = create_sawtooth(self.x_st_plot, *self.fitp1)

    def sawtooth_two(self, fit=True):
        if fit:
            self.st2, self.fitp2 = fit_sawtooth(
                self.x_shift, self.fixed_st1, period=200, amp=15
            )
        else:
            self.st2 = create_sawtooth(self.x_shift, *self.fitp2)

        self.corrected_window = (self.fixed_st1 - self.st2) + self.fitp2[2]

        # Sawtooth y values for plotting
        self.y_st2 = create_sawtooth(self.x_st_plot, *self.fitp2)

    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)


if __name__ == "__main__":

    # Mounted dataset
    print("Loaded data")
    dataset = "/home/ian/dev/bp-estimation/data/peaks/"
    pats = Pats(dataset, verbose=True)
