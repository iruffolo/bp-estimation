import sys

import numpy as np
import pandas as pd
from atriumdb import AtriumSDK, DatasetDefinition
from beat_matching import beat_matching, calculate_pat, peak_detect, rpeak_detect_fast
from gui2 import ApplicationWindow
from matplotlib.backends.qt_compat import QtWidgets
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

        # Window size in nanoseconds
        self.window_size_nano = 60 * 60 * (10**9)

        # Gap tolerance in nanoseconds
        self.gap_tol_nano = 10 * 60 * (10**9)

        self.ssize = 6

        # print_all_measures(sdk)
        self.devices = list(sdk.get_all_devices().keys())
        self.print(f"Available Devices: {self.devices}")

        self.dev = self.devices[0]

        # Initalise the device list
        self.app.add_devices(self.devices, self.device_change_cb, self.dev)

    def _update_patient_list(self):
        """
        Get the list of patients for a given device
        """

        iarr = get_ppg_ecg_intervals_device(self.sdk, self.dev, self.gap_tol_nano)

        # Filter patients that have valid overlapping ecg/ppg data
        valid_patients = []
        for i in iarr:
            map = sdk.get_device_patient_data(
                device_id_list=[self.dev], start_time=i[0], end_time=i[1]
            )
            self.print(f"Dev patient data: {map}")

            if map:
                for m in map:
                    valid_patients.append(m[1])

            if len(valid_patients) > 10:
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

    def update_patient_data(self):
        """
        Get new patient data from database, calculate inital beat matching and
        PATs.
        """

        print("Getting raw data")
        ppg, ecg, ppg_freq, ecg_freq = get_ppg_ecg_data(
            self.sdk,
            pid=self.pid,
            dev=self.dev,
            gap_tol=self.gap_tol_nano,
            window=self.window_size_nano,
        )

        # Remove offset from times
        ppg["times"] = ppg["times"] - ppg["times"][0]
        ecg["times"] = ecg["times"] - ecg["times"][0]

        print("Calculating Peaks")
        self.ecg_peak_times = rpeak_detect_fast(ecg["times"], ecg["values"], ecg_freq)
        self.ppg_peak_times = peak_detect(ppg["times"], ppg["values"], ppg_freq)

        app.add_raw_data(ppg, ecg, self.ppg_peak_times, self.ecg_peak_times)

        self.matching_beats = beat_matching(
            self.ecg_peak_times, self.ppg_peak_times, ssize=self.ssize
        )
        # print(f"Matching Peaks: {self.matching_beats}")

        # euc = [m.distance[m.n_peaks] for m in self.matching_beats]
        # for e in euc:
        #     print(e)
        # euc = (np.array(euc) - np.min(euc)) / (np.max(euc) - np.min(euc))

        cmap = [m.confidence / m.distance[m.n_peaks] for m in self.matching_beats]
        cmap = (np.array(cmap) - np.min(cmap)) / (np.max(cmap) - np.min(cmap))
        print(f"Confidence: {cmap}")
        for c in cmap:
            print(c)

        # Calculate PAT for each matched peak
        self.pats = {
            "times": np.zeros([len(self.matching_beats)]),
            "values": np.zeros([len(self.matching_beats)]),
            "confidence": np.zeros([len(self.matching_beats)]),
        }

        for i, m in enumerate(self.matching_beats):
            pat = (
                self.ppg_peak_times[m.nearest_ppg_peak + m.n_peaks]
                - self.ecg_peak_times[m.ecg_peak]
            )

            self.pats["times"][i] = self.ecg_peak_times[m.ecg_peak]
            self.pats["values"][i] = pat
            self.pats["confidence"][i] = cmap[i]

        self.c_pats = self.correct_pats()

        app.add_pat_data(self.pats, self.c_pats, cmap)

    def correct_pats(self, pat_range=0.100):
        """
        Correct PATs that are outside the expected range
        """

        c_pats_df = pd.DataFrame(self.pats)

        # Use median as expected value to correct outliers (i.e. mismatched beats)
        expected = np.median(c_pats_df["values"][c_pats_df["confidence"] > 0.7])
        print(f"Expected PAT: {expected}")

        tobedeleted = []

        for i, pat in enumerate(c_pats_df["values"]):

            # PAT outside expected range, correct it
            if not (expected - pat_range < pat < expected + pat_range):

                m = self.matching_beats[i]

                best = 0
                for s in range(self.ssize):
                    new_pat = (
                        self.ppg_peak_times[m.nearest_ppg_peak + s]
                        - self.ecg_peak_times[m.ecg_peak]
                    )

                    # Best PAT is selected as closest to expected value over search
                    if abs(new_pat - expected) < abs(best - expected):
                        best = new_pat

                if expected - pat_range < best < expected + pat_range:
                    c_pats_df.loc[i, "values"] = best

                # Couldn't find a good enough correction, remove point
                else:
                    # print(f"Cant find correction, destroying {pat}")
                    tobedeleted.append(i)

        c_pats_df = c_pats_df.drop(tobedeleted)

        return c_pats_df

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
