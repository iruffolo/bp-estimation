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
        self.dev = 0

        ## Params for stepping through dynamic plots
        self.index = 0
        self.step = 20 * 60  # 10 minutes
        self.window_s = 10 * 60  # 30 minutes
        self.points = 1000
        self.start_time = 0

        self._setup_ui()

    def _setup_ui(self):

        layout = QtWidgets.QGridLayout(self._main)

        canvas = [FigureCanvas(Figure(figsize=(5, 3))) for _ in range(6)]

        ### Row 0 - Beat Matching display
        layout.addWidget(NavigationToolbar(canvas[0], self), 0, 0, 1, 6)
        layout.addWidget(canvas[0], 1, 0, 1, 6)

        # Dropdowns for device and patient selection
        vbutton_layout = QtWidgets.QVBoxLayout()
        layout.addLayout(vbutton_layout, 1, 6, 1, 2)
        self.device_dropdown = QtWidgets.QComboBox()
        self.device_dropdown.setFixedSize(80, 30)
        self.patient_dropdown = QtWidgets.QComboBox()
        self.patient_dropdown.setFixedSize(80, 30)
        vbutton_layout.addWidget(self.device_dropdown)
        vbutton_layout.addWidget(self.patient_dropdown)

        ### Row 1 - PAT Series
        layout.addWidget(NavigationToolbar(canvas[1], self), 2, 0, 1, 6)
        layout.addWidget(canvas[1], 3, 0, 1, 6)

        button_layout = QtWidgets.QVBoxLayout()
        layout.addLayout(button_layout, 3, 6, 1, 2)

        ### Row 2 - Sawtooths
        layout.addWidget(NavigationToolbar(canvas[2], self), 4, 0)
        layout.addWidget(canvas[2], 5, 0)

        layout.addWidget(NavigationToolbar(canvas[3], self), 4, 1)
        layout.addWidget(canvas[3], 5, 1)

        layout.addWidget(NavigationToolbar(canvas[4], self), 4, 2)
        layout.addWidget(canvas[4], 5, 2)

        self.text1 = QtWidgets.QLineEdit()
        layout.addWidget(self.text1, 6, 0)
        self.text2 = QtWidgets.QLineEdit()
        layout.addWidget(self.text2, 6, 1)

        # Buttons to step through dynamic plots and change window params
        gbutton_layout = QtWidgets.QGridLayout()
        layout.addLayout(gbutton_layout, 5, 6, 1, 2)

        step_button = QtWidgets.QPushButton("Step")
        step_button.setFixedSize(50, 30)
        step_button.clicked.connect(lambda: self._step_update(step=True))
        gbutton_layout.addWidget(step_button, 0, 0)

        reset_button = QtWidgets.QPushButton("Reset")
        reset_button.setFixedSize(50, 30)
        reset_button.clicked.connect(self._reset_dynamic)
        gbutton_layout.addWidget(reset_button, 0, 1)

        # Text input for window size
        textin = QtWidgets.QLineEdit()
        textin.setFixedSize(40, 30)
        textin.textChanged.connect(self._window_size_change)
        textin.setText(f"{self.window_s / 60}")
        gbutton_layout.addWidget(textin, 0, 2)

        # Text input for ST params for both sawtooths
        p1b = QtWidgets.QPushButton("Period")
        p1b.setFixedSize(50, 30)
        gbutton_layout.addWidget(p1b, 1, 0)

        self.p1 = QtWidgets.QLineEdit()
        self.p1.setFixedSize(50, 30)
        self.p1.textChanged.connect(lambda x: self._st_param_change(x, 1, 0))
        gbutton_layout.addWidget(self.p1, 1, 1)

        self.p2 = QtWidgets.QLineEdit()
        self.p2.setFixedSize(50, 30)
        self.p2.textChanged.connect(lambda x: self._st_param_change(x, 1, 1))
        gbutton_layout.addWidget(self.p2, 1, 2)

        ph1b = QtWidgets.QPushButton("Phase")
        ph1b.setFixedSize(50, 30)
        gbutton_layout.addWidget(ph1b, 2, 0)

        self.ph1 = QtWidgets.QLineEdit()
        self.ph1.setFixedSize(50, 30)
        self.ph1.textChanged.connect(lambda x: self._st_param_change(x, 3, 0))
        gbutton_layout.addWidget(self.ph1, 2, 1)

        self.ph2 = QtWidgets.QLineEdit()
        self.ph2.setFixedSize(50, 30)
        self.ph2.textChanged.connect(lambda x: self._st_param_change(x, 3, 1))
        gbutton_layout.addWidget(self.ph2, 2, 2)

        a1b = QtWidgets.QPushButton("Amp")
        a1b.setFixedSize(50, 30)
        gbutton_layout.addWidget(a1b, 3, 0)

        self.a1 = QtWidgets.QLineEdit()
        self.a1.setFixedSize(50, 30)
        self.a1.textChanged.connect(lambda x: self._st_param_change(x, 0, 0))
        gbutton_layout.addWidget(self.a1, 3, 1)

        self.a2 = QtWidgets.QLineEdit()
        self.a2.setFixedSize(50, 30)
        self.a2.textChanged.connect(lambda x: self._st_param_change(x, 0, 1))
        gbutton_layout.addWidget(self.a2, 3, 2)

        ### Row 3 - Corrected PAT Series
        layout.addWidget(NavigationToolbar(canvas[5], self), 7, 0, 1, 2)
        layout.addWidget(canvas[5], 8, 0, 1, 6)

        # Create subplots for each canvas
        self._beat_matching_ax = canvas[0].figure.subplots()
        self._pat_series_ax = canvas[1].figure.subplots()
        self._sawtooth_ax1 = canvas[2].figure.subplots()
        self._sawtooth_ax2 = canvas[3].figure.subplots()
        self._corrected_ax = canvas[4].figure.subplots()
        self._final_ax = canvas[5].figure.subplots()

        # Call this at end after all buttons etc are created
        self._setup_plots()

    def _setup_plots(self):
        self._beat_matching_ax.set_xlabel("Time (s)")
        self._beat_matching_ax.set_ylabel("PAT (s)")
        self._beat_matching_ax.set_title(
            f"Patient: {self.pid}\n"
            f"Admit Date: {self.date.day}-{self.date.month}-{self.date.year}"
        )
        self._beat_matching_ax.grid(True)

        self._pat_series_ax.set_xlabel("Time (s)")
        self._pat_series_ax.set_ylabel("PAT (s)")
        self._pat_series_ax.set_title("Pulse Arrival Time")
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

        self._corrected_ax.set_xlabel("Time (s)")
        self._corrected_ax.set_ylabel("PAT (s)")
        self._corrected_ax.set_title("Corrected Window")
        self._corrected_ax.grid(True)

        self._final_ax.set_xlabel("Time (s)")
        self._final_ax.set_ylabel("PAT (s)")
        self._final_ax.set_title("Corrected Pulse Arrival Times")
        self._final_ax.grid(True)

        self._beat_matching_ax.sharex(self._pat_series_ax)

        self._sawtooth_ax1.sharex(self._sawtooth_ax2)
        self._sawtooth_ax2.sharex(self._corrected_ax)
        self._sawtooth_ax1.sharey(self._sawtooth_ax2)
        self._sawtooth_ax2.sharey(self._corrected_ax)

        self._beat_matching_ax.figure.tight_layout()
        self._pat_series_ax.figure.tight_layout()
        self._sawtooth_ax1.figure.tight_layout()
        self._sawtooth_ax2.figure.tight_layout()
        self._corrected_ax.figure.tight_layout()
        self._final_ax.figure.tight_layout()

        self._beat_matching_ax.figure.canvas.draw()
        self._pat_series_ax.figure.canvas.draw()
        self._sawtooth_ax1.figure.canvas.draw()
        self._sawtooth_ax2.figure.canvas.draw()
        self._corrected_ax.figure.canvas.draw()
        self._final_ax.figure.canvas.draw()

    def add_patients(self, patients, callback, select_patient=None):
        self.patient_dropdown.clear()

        print("Adding Patients")
        for p in sorted(patients):
            self.patient_dropdown.addItem(f"{int(p)}")

        self.patient_cb = callback
        self.patient_dropdown.activated.connect(self._patient_change)

        if select_patient:
            self.patient_dropdown.setCurrentText(str(select_patient))

        callback(self.patient_dropdown.currentText())

    def add_devices(self, devices, callback, select_device=None, select_patient=None):
        self.device_dropdown.clear()

        print("Adding Devices")
        for d in sorted(devices):
            self.device_dropdown.addItem(f"{int(d)}")

        self.device_cb = callback
        self.device_dropdown.activated.connect(self._device_change)

        if select_device:
            self.device_dropdown.setCurrentText(str(select_device))

        callback(self.device_dropdown.currentText(), select_patient)

    def _device_change(self, device_id):
        # Callback to update device data
        self.dev = int(self.device_dropdown.currentText())
        self.device_cb(self.dev)

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

    def _st_param_change(self, val, param, st):
        """
        Callback for changing sawtooth parameters

        val: str, new value of parameter
        param: int, which parameter to change. 0=amp, 1=period, 2=phase
        st: int, which sawtooth to change
        """

        try:  # Ensure input is a number
            new_val = float(val)
            if new_val > 0 or param == 3 or param == 0:

                if st == 0:
                    self.fitp1[param] = new_val
                    # Recalculate the sawtooths with new params
                    self._update_sawtooth1(fit=False)
                    self._update_sawtooth2(fit=True)
                elif st == 1:
                    self.fitp2[param] = new_val
                    # Recalculate the sawtooths with new params
                    self._update_sawtooth2(fit=False)

                self._update_sawtooth3()
                self._update_sawtooth_plots()
        except ValueError:
            return

    def update_patient_data(self, times, pats, date):
        # Scale x-axis to start at 0
        self.x = times - times.iloc[0]
        self.y = pats

        self.x_shift = self.x - self.x.iloc[0]
        self.x_shift_scaled = self.x_shift / self.x_shift.iloc[-1]

        self.start_time = self.x.iloc[0]

        self.date = date

        self._plot_static()
        self._plot_dynamic()
        self._setup_plots()
        self._update_text()

    def _reset_dynamic(self):
        self.start_time = self.x.iloc[0]
        self._step_update(step=False)

    def _step_update(self, step=True):
        self._update_window(step=step)
        self._update_sawtooth1()
        self._update_sawtooth2()
        self._update_sawtooth3()
        self._update_sawtooth_plots()
        self._update_text()

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

        self.xdata_shift = self.xdata - self.xdata.iloc[0]
        self.xdata_shift_scaled = self.xdata_shift / self.xdata_shift.iloc[-1]

    def _update_sawtooth1(self, fit=True):
        """
        Fit Sawtooth functions
        """

        # First Sawtooth
        if fit:
            self.st1, self.fitp1 = fit_sawtooth(
                self.xdata_shift, self.ydata, period_sec=60
            )
            # Optimize the first st
            # y_st1 = _create_sawtooth(self.xdata_shift_scaled, *self.fitp1)

            # phase_shifts = np.linspace(0, 2 * np.pi, num=314 * 2)
            # best = self.fitp1[3]
            # for ps in phase_shifts:
            #     y_st = _create_sawtooth(self.xdata_shift_scaled, *self.fitp1[:3], ps)
            #     e = sawtooth_error(self.ydata, y_st)
            #     # print(ps, e, best)
            #     if e < sawtooth_error(self.ydata, y_st1):
            #         best = ps
            #
            # print(f"Old phase: {self.fitp1[3]}")
            # print(f"New phase: {best}")
            #
            # # Update the phase of sawtooth
            # self.fitp1[3] = best

        else:
            self.st1 = _create_sawtooth(self.xdata_shift_scaled, *self.fitp1)

        self.fixed_st1 = (self.ydata - self.st1) + self.fitp1[2]

    def _update_sawtooth2(self, fit=True):
        # Second Sawtooth

        if fit:
            self.st2, self.fitp2 = fit_sawtooth(
                self.xdata_shift, self.fixed_st1, period_sec=125
            )
        else:
            self.st2 = _create_sawtooth(self.xdata_shift_scaled, *self.fitp2)

    def _update_sawtooth3(self):
        # Final corrected PATs - Both STs applied
        self.final = self.fixed_st1 - self.st2 + self.fitp2[2]

        # Apply both sawtooths to the entire series
        self.final_st1 = _create_sawtooth(self.x_shift_scaled, *self.fitp1)
        self.final_st2 = _create_sawtooth(self.x_shift_scaled, *self.fitp2)

        self.final_series = self.y - self.final_st1 + self.fitp1[2]

        # self.final_series = f1 - st2 + self.fitp2[2]

    def _update_sawtooth_plots(self):

        print(f"Updating sawtooth plot, starting at index {self.index}")

        x_ls = np.linspace(min(self.xdata_shift), max(self.xdata_shift), num=500)
        x_st = x_ls / x_ls[-1]

        # Sawtooth y values for plotting
        y_st1 = _create_sawtooth(x_st, *self.fitp1)
        y_st2 = _create_sawtooth(x_st, *self.fitp2)

        # Plot Raw data
        self._a1_p1.set_data(self.xdata, self.ydata)
        self._a2_p1.set_data(self.xdata, self.fixed_st1)
        self._a3_p1.set_data(self.xdata, self.final)

        # self._a1_p2.set_data(self.xdata, self.st1)
        self._a1_p2.set_data(x_ls + self.xdata.iloc[0], y_st1)
        self._a2_p2.set_data(x_ls + self.xdata.iloc[0], y_st2)

        self._a4_p1.set_data(self.x, self.final_series)

        # self._final_ax.plot(self.x, self.final_st1, "--", color="red")

        # Adjust the axes to follow the sawtooth.
        self._sawtooth_ax1.set_xlim(np.min(self.xdata), np.max(self.xdata))
        self._sawtooth_ax1.set_ylim(
            np.median(self.ydata) - 0.05, np.median(self.ydata) + 0.05
        )

        # Update the figure.
        self._a1_p1.figure.canvas.draw()
        self._a1_p2.figure.canvas.draw()
        self._a1_p3.figure.canvas.draw()
        self._a2_p1.figure.canvas.draw()
        self._a2_p2.figure.canvas.draw()
        self._a3_p1.figure.canvas.draw()
        self._a4_p1.figure.canvas.draw()

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

        self._pat_series_ax.sharex(self._final_ax)
        self._pat_series_ax.sharey(self._final_ax)

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

        self._a3_p1 = self._corrected_ax.plot([], [], ".", markersize=1.0)[0]

        self._a4_p1 = self._final_ax.plot([], [], ".", markersize=1.0, color="green")[0]

        self._step_update(step=False)

    def _update_text(self):
        self.p1.setText(f"{self.fitp1[1]:.3f}")
        self.p2.setText(f"{self.fitp2[1]:.3f}")
        self.ph1.setText(f"{self.fitp1[3]:.3f}")
        self.ph2.setText(f"{self.fitp2[3]:.3f}")
        self.a1.setText(f"{self.fitp1[0]:.4f}")
        self.a2.setText(f"{self.fitp2[0]:.4f}")

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


if __name__ == "__main__":
    # path = "/home/ian/dev/bp-estimation/data/paper_results/"
    path = "/home/ian/dev/bp-estimation/data/paper_results/"
    files = os.listdir(path)
    pats_fn = [f for f in files if f"_pats.csv" in f]
    df = pd.read_csv(f"{path}/{pats_fn[10]}")
    pids = df["patient_id"].unique()

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

    print(pats_fn[10])
    print(pats_fn[10])
    print(pats_fn[10])
    print(pats_fn[10])
    print(pats_fn[10])

    pats = df[df["patient_id"] == pids[1]]

    app.update_patient_data(pats["ecg_peaks"], pats["pat"])
    app.plot_dynamic(pats["ecg_peaks"], pats["pat"])

    app.show()
    app.activateWindow()
    app.raise_()
    qapp.exec()
