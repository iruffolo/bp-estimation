import os
import sys
import time

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.figure import Figure
from sawtooth import _create_sawtooth, fit_sawtooth


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self, argv):
        super().__init__()

        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)

        layout = QtWidgets.QGridLayout(self._main)

        # Main canvas for entire PAT series
        static_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        layout.addWidget(NavigationToolbar(static_canvas, self), 0, 0, 1, 2)
        layout.addWidget(static_canvas, 1, 0, 1, 2)

        # patient_dropdown = QComboBox()

        dynamic_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        layout.addWidget(NavigationToolbar(dynamic_canvas, self), 2, 0)
        layout.addWidget(dynamic_canvas, 3, 0)
        self.text1 = QtWidgets.QLineEdit()
        layout.addWidget(self.text1, 4, 0)

        dynamic_canvas2 = FigureCanvas(Figure(figsize=(5, 3)))
        layout.addWidget(NavigationToolbar(dynamic_canvas2, self), 2, 1)
        layout.addWidget(dynamic_canvas2, 3, 1)
        self.text2 = QtWidgets.QLineEdit()
        layout.addWidget(self.text2, 4, 1)

        button = QtWidgets.QPushButton("Step")
        button.setFixedSize(100, 60)
        button.clicked.connect(self._update_dynamic_data)
        layout.addWidget(button, 5, 0)

        final_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        layout.addWidget(NavigationToolbar(final_canvas, self), 6, 0, 1, 2)
        layout.addWidget(final_canvas, 7, 0, 1, 2)

        # Create canvas for static and dynamic plots
        self._pat_series_ax = static_canvas.figure.subplots()
        self._sawtooth_ax1 = dynamic_canvas.figure.subplots()
        self._sawtooth_ax2 = dynamic_canvas2.figure.subplots()
        self._final_ax = final_canvas.figure.subplots()
        self._setup_plots()

        # Params for stepping through dynamic plots
        self.index = 0
        self.step = 10 * 60  # 10 minutes
        self.window_s = 10 * 60  # 10 minutes
        self.points = 1000

        self.start_time = 0

    def _setup_plots(self):
        self._pat_series_ax.set_xlabel("Time (s)")
        self._pat_series_ax.set_ylabel("PAT (s)")
        self._pat_series_ax.set_title("Patient X Series")
        self._pat_series_ax.grid(True)

        self._sawtooth_ax1.set_xlabel("Time (s)")
        self._sawtooth_ax1.set_ylabel("PAT (s)")
        self._sawtooth_ax1.set_title("First sawtooth fit - 10 min windows")
        self._sawtooth_ax1.grid(True)

        self._sawtooth_ax2.set_xlabel("Time (s)")
        self._sawtooth_ax2.set_ylabel("PAT (s)")
        self._sawtooth_ax2.set_title("Second sawtooth fit - 10 min windows")
        self._sawtooth_ax2.grid(True)

        self._final_ax.set_xlabel("Time (s)")
        self._final_ax.set_ylabel("PAT (s)")
        self._final_ax.set_title("Final Corrected PATs")
        self._final_ax.grid(True)

    def update_patient_data(self, times, pats):
        self.x = times
        self.y = pats

        self.start_time = self.x.iloc[0]

        self._plot_static()
        self._plot_dynamic()
        self._setup_plots()

    def _plot_static(self):
        self._pat_series_ax.clear()

        self._pat_series_ax.plot(self.x, self.y, ".", markersize=1.0)
        median = np.median(self.y)
        yrange = np.std(self.y) * 3
        self._pat_series_ax.set_ylim(median - yrange, median + yrange)

        self.st, self.fitp = fit_sawtooth(self.x, self.y, plot=False)
        x_st = np.linspace(min(self.x), max(self.x), num=len(self.x))
        y_st = _create_sawtooth(x_st, *self.fitp)
        self._pat_series_ax.plot(x_st, y_st, "--", color="orange")

    def _plot_dynamic(self):
        self._sawtooth_ax1.clear()
        self._sawtooth_ax2.clear()
        self._final_ax.clear()

        # self._sawtooth_ax1.sharex(self._sawtooth_ax2)
        # self._sawtooth_ax1.sharey(self._sawtooth_ax2)

        self._a1_p1 = self._sawtooth_ax1.plot([], [], ".", markersize=1.0)[0]
        self._a1_p2 = self._sawtooth_ax1.plot(
            [], [], "--", markersize=1.0, color="red"
        )[0]
        self._a1_p3 = self._sawtooth_ax1.plot([], [], "--", markersize=1.0)[0]

        self._a2_p1 = self._sawtooth_ax2.plot([], [], ".", markersize=1.0)[0]
        self._a2_p2 = self._sawtooth_ax2.plot(
            [], [], "--", markersize=1.0, color="red"
        )[0]

        self._a3_p1 = self._final_ax.plot([], [], ".", markersize=1.0)[0]

        self._update_dynamic_data()

    def _update_dynamic_data(self):

        print(f"Updating sawtooth, starting at index {self.index}")

        # Shift the pats to the left based on window size
        end_time = self.start_time + self.window_s

        idx = np.where((self.x > self.start_time) & (self.x < end_time))[0]
        xdata = self.x.iloc[idx]
        ydata = self.y.iloc[idx]

        # Step start time for next iteration
        self.start_time += self.step

        # Adjust the axes to follow the sawtooth.
        self._sawtooth_ax1.set_xlim(np.min(xdata), np.max(xdata))
        self._sawtooth_ax1.set_ylim(np.min(ydata), np.max(ydata))
        self._sawtooth_ax2.set_xlim(np.min(xdata), np.max(xdata))
        self._sawtooth_ax2.set_ylim(np.min(ydata), np.max(ydata))
        self._final_ax.set_xlim(np.min(xdata), np.max(xdata))
        self._final_ax.set_ylim(np.min(ydata), np.max(ydata))

        # Raw data
        self._a1_p1.set_data(xdata, ydata)
        st_points = 200

        # Display sawtooth fit on entire patient for comparison
        # orig_x_st = np.linspace(min(xdata), max(xdata), num=st_points)
        # orig_y_st = _create_sawtooth(orig_x_st, *self.fitp)
        # self._a1_p3.set_data(orig_x_st, orig_y_st)

        st, fitp = fit_sawtooth(xdata, ydata, plot=False)
        x_st = np.linspace(min(xdata), max(xdata), num=st_points)
        y_st = _create_sawtooth(x_st, *fitp)
        self._a1_p2.set_data(x_st, y_st)
        p = (2 * np.pi) / fitp[1]
        self.text1.setText(
            f"Sawtooth 1 fit: A={fitp[0]:.4f}, Fi={fitp[1]:.2f}, offset={fitp[2]:.2f}"
        )

        fixed_st1 = ydata - st + fitp[2]
        self._a2_p1.set_data(xdata, fixed_st1)

        st, fitp = fit_sawtooth(xdata, fixed_st1, plot=False)
        x_st = np.linspace(min(xdata), max(xdata), num=st_points)
        y_st = _create_sawtooth(x_st, *fitp)
        self._a2_p2.set_data(x_st, y_st)
        p = (2 * np.pi) / fitp[1]
        self.text2.setText(
            f"Sawtooth 2 fit: A={fitp[0]:.4f}, Fi={fitp[1]:.2f}, offset={fitp[2]:.2f}"
        )

        final = fixed_st1 - st + fitp[2]
        self._a3_p1.set_data(xdata, final)

        # Update the figure.
        self._a1_p1.figure.canvas.draw()
        self._a1_p2.figure.canvas.draw()
        self._a1_p3.figure.canvas.draw()
        self._a2_p1.figure.canvas.draw()
        self._a2_p2.figure.canvas.draw()
        self._a3_p1.figure.canvas.draw()
        # self._p2.figure.canvas.draw()
        # self._p3.figure.canvas.draw()


if __name__ == "__main__":
    # Check whether there is already a running QApplication (e.g., if running
    # from an IDE).
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)
    app = ApplicationWindow()

    path = "/home/ian/dev/bp-estimation/data/paper_results_short/"
    # path = "/home/ian/dev/bp-estimation/data/paper_results_short/"
    files = os.listdir(path)
    pats_fn = [f for f in files if f"_pats.csv" in f]
    df = pd.read_csv(f"{path}/{pats_fn[10]}")
    pids = df["patient_id"].unique()

    pats = df[df["patient_id"] == pids[1]]

    app.update_patient_data(pats["ecg_peaks"], pats["pat"])
    app.plot_dynamic(pats["ecg_peaks"], pats["pat"])

    app.show()
    app.activateWindow()
    app.raise_()
    qapp.exec()
