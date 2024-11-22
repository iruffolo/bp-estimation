import sys
from datetime import datetime

import numpy as np
import pandas as pd
from atriumdb import AtriumSDK, DatasetDefinition
from beat_matching import beat_matching, correct_pats, peak_detect, rpeak_detect_fast
from gui2 import ApplicationWindow
from matplotlib.backends.qt_compat import QtWidgets
from new_st import create_sawtooth, fit_sawtooth
from utils.atriumdb_helpers import (
    get_ppg_ecg_data,
    get_ppg_ecg_intervals_device,
    print_all_measures,
)


class DataManager:
    def __init__(self, sdk: AtriumSDK, app: ApplicationWindow, verbose=False):
        self.sdk = sdk
        self.app = app

        self.verbose = verbose

        # Window and gap size for data pull
        self.window_size_nano = 60 * 60 * (10**9)
        self.gap_tol_nano = 10 * 60 * (10**9)

        self.ssize = 6

        ## Params for stepping through dynamic plots
        self.index = 0
        self.step = 20 * 60  # 10 minutes
        self.window_s = 20 * 60  # 30 minutes
        self.points = 1000
        self.start_time = 0

        # print_all_measures(sdk)
        self.devices = list(sdk.get_all_devices().keys())
        self.print(f"Available Devices: {self.devices}")

        self.dev = self.devices[0]

        # Initalise the device list
        self.app.add_devices(self.devices, self.device_change_cb, self.dev)

        self.app.set_button_callbacks(
            self.step_cb,
            self.reset_cb,
            self.window_change_cb,
            self.sawtooth_cb,
            self.update_cb,
        )

    def _update_patient_list(self):
        """
        Get the list of patients for a given device
        """

        iarr = get_ppg_ecg_intervals_device(self.sdk, self.dev, self.gap_tol_nano)

        # Filter patients that have valid overlapping self.ecg/self.ppg data
        valid_patients = []
        for i in iarr:
            map = sdk.get_device_patient_data(
                device_id_list=[self.dev], start_time=i[0], end_time=i[1]
            )
            self.print(f"Dev patient data: {map}")

            if map:
                for m in map:
                    valid_patients.append(m[1])

            if len(valid_patients) > 50:
                break

        self.patients = np.unique(valid_patients)
        self.pid = self.patients[9]

        self.app.add_patients(self.patients, self.patient_change_cb, self.pid)

        self.print(f"Available Patients: {self.patients}")

    def patient_change_cb(self, patient_id):
        self.print(f"Patient changed {patient_id}")
        self.pid = int(patient_id)

        self.update_patient_data()

    def device_change_cb(self, device_id):
        self.print(f"Device changed {device_id}")
        self.dev = int(device_id)

        # Update patient list and select the first patient
        self._update_patient_list()
        self.patient_change_cb(self.pid)

    def step_cb(self):
        print("Step cb")
        pass
        return

    def reset_cb(self):
        print("Reset cb")
        # self.start_time = self.x.iloc[0]
        # self._step_update(step=False)

    def update_cb(self):
        self.update_sawtooth_plots()

    def window_change_cb(self, window_size):
        # Only update if window is a valid int
        try:
            new_window = int(window_size) * 60
            if new_window <= 0:
                new_window = 10 * 60
            print(f"Window size changed to {window_size}")
            self.window_s = new_window
            self.update_window()
            self.sawtooth_one()
            self.sawtooth_two()
        except ValueError:
            return

    def sawtooth_cb(self, value, param, st):
        try:  # Ensure input is a number
            new_val = float(value)
            if new_val > 0 or param == 3 or param == 0:

                if st == 0:
                    self.fitp1[param] = new_val
                    self.sawtooth_one(fit=False)
                    self.sawtooth_two(fit=True)
                elif st == 1:
                    self.fitp2[param] = new_val
                    self.sawtooth_two(fit=False)

        except ValueError:
            return

    def update_patient_data(self):
        """
        Get new patient data from database, calculate inital beat matching and
        PATs.
        """

        print("Getting raw data")
        self.ppg, self.ecg, self.ppg_freq, self.ecg_freq = get_ppg_ecg_data(
            self.sdk,
            pid=self.pid,
            dev=self.dev,
            gap_tol=self.gap_tol_nano,
            window=self.window_size_nano,
        )
        self.date = datetime.fromtimestamp(self.ecg["times"][0])
        self.app.set_title(f"Patient {self.pid} Date: {self.date}")

        # Remove offset from times
        self.ppg["times"] = self.ppg["times"] - self.ppg["times"][0]
        self.ecg["times"] = self.ecg["times"] - self.ecg["times"][0]

        self.print("Calculating Peaks")
        self.ecg_peak_times = rpeak_detect_fast(
            self.ecg["times"], self.ecg["values"], self.ecg_freq
        )
        self.ppg_peak_times = peak_detect(
            self.ppg["times"], self.ppg["values"], self.ppg_freq
        )

        self.print("Adding raw data to GUI")
        self.app.plot_raw_data(
            self.ppg, self.ecg, self.ppg_peak_times, self.ecg_peak_times
        )

        # Calculate beat matching and PATs
        self.beat_matching()

        # Plot all the results
        self.app.plot_beat_match_data(self.all_pats)
        self.app.plot_pat_data(self.pats, self.c_pats_df, self.cmap)

        # Create smaller window slice for sawtooth fitting
        self.start_time = 0
        self.update_window()

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

    qapp = QtWidgets.QApplication(sys.argv)
    app = ApplicationWindow(sys.argv)

    # Mounted dataset
    local_dataset = "/home/ian/dev/datasets/ian_dataset_2024_08_26"

    sdk = AtriumSDK(dataset_location=local_dataset)

    dm = DataManager(sdk, app, verbose=True)

    app.show()
    app.activateWindow()
    app.raise_()
    qapp.exec()
