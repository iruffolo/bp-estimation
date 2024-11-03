import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib.backends.qt_compat import QtWidgets
from qtgui import ApplicationWindow
from sawtooth import _create_sawtooth, fit_sawtooth


class DataManger:
    def __init__(self, path):
        self.path = path
        self.files = os.listdir(path)
        self.df = None

    def patient_change_callback(self, patient_id):
        print(f"Patient changed {patient_id}")
        pid = int(patient_id)

        pats = self.df[self.df["patient_id"] == pid]

        x = pats["ecg_peaks"]
        y = pats["pat"]

        date = datetime.fromtimestamp(pats["ecg_peaks"].iloc[0])
        app.update_patient_data(x, y, date)

    def device_change_callback(self, device_id):
        print(f"Device changed {device_id}")
        did = int(device_id)

        pats_fn = [f for f in self.files if f"{did}_pats.csv" in f][0]

        self.df = pd.read_csv(f"{path}/{pats_fn}")

        pids = self.df["patient_id"].unique()
        app.add_patients(pids, self.patient_change_callback)


if __name__ == "__main__":

    qapp = QtWidgets.QApplication(sys.argv)
    app = ApplicationWindow(sys.argv)

    # path = "/home/ian/dev/bp-estimation/data/paper_results_short/"
    path = "/home/ian/dev/bp-estimation/data/paper_results/"

    dm = DataManger(path)
    devices = list(range(74, 116))

    app.add_devices(devices, dm.device_change_callback)

    app.show()
    app.activateWindow()
    app.raise_()
    qapp.exec()
