import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.figure import Figure
from new_st import _create_sawtooth, fit_sawtooth, sawtooth_error


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self, argv):
        super().__init__()

        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)

        self.date = datetime.fromisocalendar(1970, 1, 1)
        self.pid = 0

        ## Params for stepping through dynamic plots
        self.index = 0
        self.step = 10 * 60  # 10 minutes
        self.window_s = 10 * 60  # 10 minutes
        self.points = 1000
        self.start_time = 0

        layout = QtWidgets.QGridLayout(self._main)

        ## Plotting Canvases

        # Main canvas for entire PAT series
        static_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        layout.addWidget(NavigationToolbar(static_canvas, self), 0, 0, 1, 2)
        layout.addWidget(static_canvas, 1, 0, 1, 2)

        # Secondary canvas for first sawtooth fit
        dynamic_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        layout.addWidget(NavigationToolbar(dynamic_canvas, self), 2, 0)
        layout.addWidget(dynamic_canvas, 3, 0)
        self.text1 = QtWidgets.QLineEdit()
        layout.addWidget(self.text1, 4, 0)

        # Secondary canvas for second sawtooth fit
        dynamic_canvas2 = FigureCanvas(Figure(figsize=(5, 3)))
        layout.addWidget(NavigationToolbar(dynamic_canvas2, self), 2, 1)
        layout.addWidget(dynamic_canvas2, 3, 1)
        self.text2 = QtWidgets.QLineEdit()
        layout.addWidget(self.text2, 4, 1)

        # Final canvas for corrected PATs with both sawtooth fits applied
        final_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        layout.addWidget(NavigationToolbar(final_canvas, self), 5, 0, 1, 2)
        layout.addWidget(final_canvas, 6, 0, 1, 2)
        ### Control Buttons on the right

        # Create canvas for static and dynamic plots
        self._pat_series_ax = static_canvas.figure.subplots()
        self._sawtooth_ax1 = dynamic_canvas.figure.subplots()
        self._sawtooth_ax2 = dynamic_canvas2.figure.subplots()
        self._final_ax = final_canvas.figure.subplots()
        self._setup_plots()

        # Dropdowns for device and patient selection
        vbutton_layout = QtWidgets.QVBoxLayout()
        layout.addLayout(vbutton_layout, 1, 2, 1, 2)
        self.device_dropdown = QtWidgets.QComboBox()
        self.device_dropdown.setFixedSize(80, 30)
        self.patient_dropdown = QtWidgets.QComboBox()
        self.patient_dropdown.setFixedSize(80, 30)
        # self.patient_dropdown.addItem("Patient X")
        vbutton_layout.addWidget(self.device_dropdown)
        vbutton_layout.addWidget(self.patient_dropdown)

        # Buttons to step through dynamic plots and change window params
        vbutton_layout1 = QtWidgets.QVBoxLayout()
        layout.addLayout(vbutton_layout1, 3, 2, 1, 2)
        step_button = QtWidgets.QPushButton("Step")
        step_button.setFixedSize(100, 50)
        step_button.clicked.connect(lambda: self._step_update(step=True))
        vbutton_layout1.addWidget(step_button)
        reset_button = QtWidgets.QPushButton("Reset")
        reset_button.setFixedSize(100, 50)
        reset_button.clicked.connect(self._reset_dynamic)
        vbutton_layout1.addWidget(reset_button)

        # Text input for window size
        textin = QtWidgets.QLineEdit()
        textin.setFixedSize(100, 30)
        textin.textChanged.connect(self._window_size_change)
        textin.setText(f"{self.window_s / 60}")
        vbutton_layout1.addWidget(textin)

        # Text input for ST params
        textin_st_p = QtWidgets.QLineEdit()
        textin_st_p.setFixedSize(100, 30)
        textin_st_p.textChanged.connect(self._st_period_change)
        textin_st_p.setText("Period")
        vbutton_layout1.addWidget(textin_st_p)

        textin_st_phase = QtWidgets.QLineEdit()
        textin_st_phase.setFixedSize(100, 30)
        textin_st_phase.textChanged.connect(self._st_phase_change)
        textin_st_phase.setText("Phase")
        vbutton_layout1.addWidget(textin_st_phase)

    def add_patients(self, patients, callback):
        self.patient_dropdown.clear()

        print("Adding Patients")
        for p in sorted(patients):
            self.patient_dropdown.addItem(f"{int(p)}")

        self.patient_cb = callback
        self.patient_dropdown.activated.connect(self._patient_change)

        callback(self.patient_dropdown.currentText())

    def add_devices(self, devices, callback):
        self.device_dropdown.clear()

        print("Adding Devices")
        for d in sorted(devices):
            self.device_dropdown.addItem(f"{int(d)}")

        self.device_cb = callback
        self.device_dropdown.activated.connect(self._device_change)

        callback(self.device_dropdown.currentText())

    def _device_change(self, device_id):
        # Callback to update device data
        self.did = int(self.device_dropdown.currentText())
        self.device_cb(self.did)

    def _patient_change(self, patient_id):
        # Callback to update patient data
        self.pid = int(self.patient_dropdown.currentText())
        self.patient_cb(self.pid)

    def _window_size_change(self, window_s):

        try:  # Check if input is a number
            new_window = int(window_s) * 60
            if new_window <= 0:
                new_window = 10 * 60
            # print(f"Window size changed to {window_s}")
            self.window_s = new_window
            self._setup_plots()
            self._reset_dynamic()
        except ValueError:
            return

    def _st_period_change(self, st_period):
        try:  # Check if input is a number
            new_st_period = float(st_period)
            if new_st_period > 0:
                self.fitp1[1] = st_period
                self._update_sawtooth1(fit=False)
                self._update_sawtooth2()
                self._update_sawtooth3()
                self._update_dynamic_plot()
        except ValueError:
            return

    def _st_phase_change(self, st_phase):
        try:  # Check if input is a number
            new_st_phase = float(st_phase)
            print(f"New phase: {new_st_phase}")

            self.fitp1[3] = st_phase
            self._update_sawtooth1(fit=False)
            self._update_sawtooth2()
            self._update_sawtooth3()
            self._update_dynamic_plot()
        except ValueError:
            return

    def update_patient_data(self, times, pats, date):
        # Scale x-axis to start at 0
        self.x = times - times.iloc[0]
        self.y = pats

        self.start_time = self.x.iloc[0]

        self.date = date

        self._plot_static()
        self._plot_dynamic()
        self._setup_plots()

    def _plot_static(self):
        print("Plotting static")
        self._pat_series_ax.clear()

        self._pat_series_ax.plot(self.x, self.y, ".", markersize=1.0)
        median = np.median(self.y)
        yrange = 0.5
        print(f"Median: {median}, Range: {yrange}")

        self.ymin = median - yrange
        self.ymax = median + yrange
        self._pat_series_ax.set_ylim(self.ymin, self.ymax)
        self._pat_series_ax.figure.canvas.draw()

    def _plot_dynamic(self):
        self._sawtooth_ax1.clear()
        self._sawtooth_ax2.clear()
        self._final_ax.clear()

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
        self._step_update(step=False)

    def _reset_dynamic(self):
        self.start_time = self.x.iloc[0]
        self._step_update(step=False)

    def _step_update(self, step=True):
        self._update_window(step=step)
        self._update_sawtooth1()
        self._update_sawtooth2()
        self._update_sawtooth3()
        self._update_dynamic_plot()

    def _update_window(self, step=True):
        if step:
            print("steppin")
            self.start_time += self.step

        # Shift the pats to the left based on window size
        end_time = self.start_time + self.window_s

        idx = np.where((self.x >= self.start_time) & (self.x <= end_time))[0]
        if not idx.size:
            print("Window out of range, restarting from beginning")
            self._reset_dynamic()
            return

        self.xdata = self.x.iloc[idx]
        self.ydata = self.y.iloc[idx]

        self.x_shift = self.xdata - self.xdata.iloc[0]

    def _update_sawtooth1(self, fit=True):
        """
        Fit Sawtooth functions
        """

        # First Sawtooth
        if fit:
            self.st1, self.fitp1 = fit_sawtooth(self.x_shift, self.ydata, period_sec=51)
            # Optimize the first st
            y_st1 = _create_sawtooth(self.x_shift, *self.fitp1)

            phase_shifts = np.linspace(0, 2 * np.pi, num=314 * 2)
            best = self.fitp1[3]
            for ps in phase_shifts:
                y_st = _create_sawtooth(self.x_shift, *self.fitp1[:3], ps)
                e = sawtooth_error(self.ydata, y_st)

                print(ps, e, best)
                if e < sawtooth_error(self.ydata, y_st1):
                    best = ps

            print(f"Best phase: {best}")
            print(f"Old phase: {self.fitp1[3]}")

            # Update the phase of sawtooth
            self.fitp1[3] = best

        else:
            self.st1 = _create_sawtooth(self.x_shift, *self.fitp1)

        self.fixed_st1 = self.ydata - self.st1 + self.fitp1[2]

    def _update_sawtooth2(self):
        # Second Sawtooth
        self.st2, self.fitp2 = fit_sawtooth(
            self.x_shift, self.fixed_st1, period_sec=125
        )

    def _update_sawtooth3(self):
        # Final corrected PATs - Both STs applied
        self.final = self.fixed_st1 - self.st2 + self.fitp2[2]

    def _update_dynamic_plot(self):

        print(f"Updating sawtooth, starting at index {self.index}")

        x_ls = np.linspace(min(self.x_shift), max(self.x_shift), num=500)
        x_st = x_ls / x_ls[-1]

        # Sawtooth y values for plotting
        y_st1 = _create_sawtooth(x_st, *self.fitp1)
        y_st2 = _create_sawtooth(x_st, *self.fitp2)

        # y_st1_best = _create_sawtooth(x_st, *self.fitp1[:3], self.best_phase)
        # self._sawtooth_ax1.plot(
        #     x_ls + self.xdata.iloc[0], y_st1_best, "--", color="green"
        # )

        # Plot Raw data
        self._a1_p1.set_data(self.xdata, self.ydata)
        self._a2_p1.set_data(self.xdata, self.fixed_st1)
        self._a3_p1.set_data(self.xdata, self.final)

        self._a1_p2.set_data(x_ls + self.xdata.iloc[0], y_st1)
        self._a2_p2.set_data(x_ls + self.xdata.iloc[0], y_st2)

        self.text1.setText(
            f"Sawtooth 1 fit: "
            f"A={self.fitp1[0]:.4f}, "
            f"Period={self.fitp1[1]:.2f}, "
            f"Offset={self.fitp1[2]:.2f}, "
            f"Phase={self.fitp1[3]:.2f}"
        )
        self.text2.setText(
            f"Sawtooth 1 fit: "
            f"A={self.fitp2[0]:.4f}, "
            f"Period={self.fitp2[1]:.2f}, "
            f"Offset={self.fitp2[2]:.2f}, "
            f"Phase={self.fitp2[3]:.2f}"
        )

        # Adjust the axes to follow the sawtooth.
        self._sawtooth_ax1.set_xlim(np.min(self.xdata), np.max(self.xdata))
        # self._sawtooth_ax1.set_ylim(
        #     np.median(self.ydata) - 0.05, np.median(self.ydata) + 0.05
        # )
        self._sawtooth_ax1.set_ylim(0, 2)

        # Update the figure.
        self._a1_p1.figure.canvas.draw()
        self._a1_p2.figure.canvas.draw()
        self._a1_p3.figure.canvas.draw()
        self._a2_p1.figure.canvas.draw()
        self._a2_p2.figure.canvas.draw()
        self._a3_p1.figure.canvas.draw()

    def _setup_plots(self):
        self._pat_series_ax.set_xlabel("Time (s)")
        self._pat_series_ax.set_ylabel("PAT (s)")
        self._pat_series_ax.set_title(
            f"Patient: {self.pid}\n"
            f"Admit Date: {self.date.day}-{self.date.month}-{self.date.year}"
        )
        self._pat_series_ax.grid(True)

        self._sawtooth_ax1.set_xlabel("Time (s)")
        self._sawtooth_ax1.set_ylabel("PAT (s)")
        self._sawtooth_ax1.set_title(
            f"First sawtooth fit - {self.window_s / 60} min windows"
        )
        self._sawtooth_ax1.grid(True)

        self._sawtooth_ax2.set_xlabel("Time (s)")
        self._sawtooth_ax2.set_ylabel("PAT (s)")
        self._sawtooth_ax2.set_title(
            f"Second sawtooth fit - {self.window_s / 60} min windows"
        )
        self._sawtooth_ax2.grid(True)

        self._final_ax.set_xlabel("Time (s)")
        self._final_ax.set_ylabel("PAT (s)")
        self._final_ax.set_title("Final Corrected PATs")
        self._final_ax.grid(True)

        self._sawtooth_ax1.sharex(self._sawtooth_ax2)
        self._sawtooth_ax2.sharex(self._final_ax)
        self._sawtooth_ax1.sharey(self._sawtooth_ax2)
        self._sawtooth_ax2.sharey(self._final_ax)

        self._pat_series_ax.figure.canvas.draw()
        self._sawtooth_ax1.figure.canvas.draw()
        self._sawtooth_ax2.figure.canvas.draw()
        self._final_ax.figure.canvas.draw()

        self._pat_series_ax.figure.tight_layout()
        self._sawtooth_ax1.figure.tight_layout()
        self._sawtooth_ax2.figure.tight_layout()
        self._final_ax.figure.tight_layout()


if __name__ == "__main__":
    # Check whether there is already a running QApplication (e.g., if running
    # from an IDE).
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)
    app = ApplicationWindow()

    # path = "/home/ian/dev/bp-estimation/data/paper_results/"
    path = "/home/ian/dev/bp-estimation/data/paper_results_short/"
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
