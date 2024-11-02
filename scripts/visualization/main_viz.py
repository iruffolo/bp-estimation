import os
import sys

import numpy as np
import pandas as pd
from matplotlib.backends.qt_compat import QtWidgets
from qtgui import ApplicationWindow
from sawtooth import _create_sawtooth, fit_sawtooth

if __name__ == "__main__":

    qapp = QtWidgets.QApplication(sys.argv)
    app = ApplicationWindow(sys.argv)

    path = "/home/ian/dev/bp-estimation/data/paper_results_short/"
    # path = "/home/ian/dev/bp-estimation/data/paper_results/"
    files = os.listdir(path)
    pats_fn = [f for f in files if f"_pats.csv" in f]
    df = pd.read_csv(f"{path}/{pats_fn[10]}")
    pids = df["patient_id"].unique()

    pats = df[df["patient_id"] == pids[1]]

    x = pats["ecg_peaks"]
    y = pats["pat"]
    app.update_patient_data(x, y)

    app.show()
    app.activateWindow()
    app.raise_()
    qapp.exec()
