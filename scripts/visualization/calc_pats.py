import concurrent.futures
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
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
    def __init__(self, datapath, savepath, verbose=False):
        self.datapath = datapath
        self.savepath = savepath

        self.verbose = verbose

        # Window size for initial data slice
        self.nrows = 10000

        self.window_size_sec = 60 * 60
        self.ssize = 6

        self.ecg_freq = 500
        self.ppg_freq = 125

        self.num_heartbeats = 0

        # Load file names from path, extract patient ids and device ids
        self.process_files()
        self.devices = self.p_df["dev"].unique()

        self.process_devices()

        self.print_stats()

    def print_stats(self):
        """
        Prints some stats about the dataset
        """
        print(f"Total patients: {self.p_df['pid'].nunique()}")
        print(f"Total devices: {self.p_df['dev'].nunique()}")
        print(f"Total beats: {self.num_heartbeats}")

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
            _, dev, pid, month, year, ftype = file.split("_")
            patients["pid"].append(int(pid))
            patients["dev"].append(int(dev))
            patients["month"].append(int(month))
            patients["year"].append(int(year))
            patients["files"].append(file)

        self.p_df = pd.DataFrame(patients)

    def process_devices(self, cores=1):
        """
        Process all devices in the dataset
        """

        self.num_heartbeats = 0
        self.process_patients(self.devices[0])

        return

        with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
            results = [executor.submit(self.process_patients, d) for d in self.devices]

            for f in concurrent.futures.as_completed(results):
                print(f.result())
                self.num_heartbeats += f.result()

    def process_patients(self, d):
        """
        Process all patients in the dataset
        """

        print(f"Processing device {d}")

        heartbeats = 0

        for p in self.p_df["pid"][self.p_df["dev"] == d].unique():

            files = self.p_df["files"][
                (self.p_df["pid"] == p) & (self.p_df["dev"] == d)
            ].values
            print(files)

            heartbeats += self.process_patient(p, d, files)

        return heartbeats

    def process_patient(self, patient, device, files):
        print(f"Processing patient:\n{patient}")

        print("Getting raw data")
        ecg_files = [f for f in files if "ecg" in f]
        ppg_files = [f for f in files if "ppg" in f]

        ecg_beats = []
        ppg_beats = []

        for ecg_file, ppg_file in zip(ecg_files, ppg_files):
            ecg = pd.read_csv(
                self.datapath + ecg_file, nrows=self.nrows
            ).values.flatten()
            ppg = pd.read_csv(
                self.datapath + ppg_file, nrows=self.nrows
            ).values.flatten()

            ecg_beats.extend(ecg)
            ppg_beats.extend(ppg)

        print(f"Total ECG: {len(ecg_beats)} Total PPG: {len(ppg_beats)}")
        num_heartbeats = len(ecg_beats)

        ecg_beats = np.array(ecg_beats)
        ppg_beats = np.array(ppg_beats)

        pats = self.beat_matching(ecg_beats, ppg_beats)
        # self.save_pats(patient, device, pats)

        self.sawtooth_one(pats)

        return num_heartbeats

    def save_pats(self, patient, device, pats):
        """
        Save the PATs to a CSV file
        """
        print(f"Saving PATs for patient {patient} device {device}")

        fn = os.path.join(self.savepath, f"{device}_{patient}.csv")
        pats.to_csv(fn, header="column_names", index=False)

    def beat_matching(self, ecg_beats, ppg_beats):
        """ """

        print("Beat Matching")
        matching_beats = beat_matching(ecg_beats, ppg_beats, ssize=self.ssize)

        # Create a df for all possible PAT values
        all_pats = pd.DataFrame(
            [m.possible_pats for m in matching_beats],
            columns=[f"{i + 1} beats" for i in range(self.ssize)],
        )

        all_pats["bm_pat"] = [m.possible_pats[m.n_peaks] for m in matching_beats]
        all_pats["confidence"] = [m.confidence for m in matching_beats]
        all_pats["times"] = [ecg_beats[m.ecg_peak] for m in matching_beats]
        all_pats["beats_skipped"] = [m.n_peaks for m in matching_beats]
        # self.all_pats.set_index("times", inplace=True)

        # Correct the PATs
        print("Correcting PATs")
        correct_pats(all_pats, matching_beats, pat_range=0.300)
        print(all_pats.head())

        # cmap = [m.confidence for m in matching_beats]
        # cmap = (np.array(cmap) - np.min(cmap)) / (np.max(cmap) - np.min(cmap))

        return all_pats

    def sawtooth_one(self, pats):

        print("Fitting sawtooth")

        pats = pats[pats["valid_correction"] > 0]

        # Shift to zero for easy plotting
        x = pats["times"].values - pats["times"].iloc[0]
        y = pats["corrected_bm_pat"]

        st, fitp = fit_sawtooth(x, y, amp=50)
        print(
            f"Fit: amplitude {fitp[0]}, period {fitp[1]}, offset {fitp[2]}, phase {fitp[3]}"
        )

        fixed_st = (y - st) + fitp[2]

        from test import piecewise

        # Sawtooth y values for plotting
        x_ls = np.linspace(min(x), max(x), num=500)
        y_st = create_sawtooth(x_ls, *fitp)
        poly = np.polyfit(x, y, deg=5)

        fig, ax = plt.subplots()

        xdist = piecewise(y=y, n=6, init="2dist")
        ax.plot(xdist, y[xdist], label="approx (dist)")
        ax.scatter(x, y, marker="x")
        ax.scatter(x, fixed_st, alpha=0.5, marker=".")
        ax.plot(x_ls, y_st, alpha=0.8, color="red")

        plt.show()


if __name__ == "__main__":

    # Mounted dataset
    print("Loaded data")
    dataset = "/home/ian/dev/bp-estimation/data/peaks_ecg_ppg/"
    savepath = "/home/ian/dev/bp-estimation/data/sawtooth/"
    pats = Pats(dataset, savepath, verbose=True)
