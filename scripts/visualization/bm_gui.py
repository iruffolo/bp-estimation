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
from new_st import create_sawtooth, fit_sawtooth, sawtooth_error
from pyhrv.nonlinear import poincare


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self, argv):
        super().__init__()

        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)

        # Create main layout for GUI
        main_layout = QtWidgets.QVBoxLayout(self._main)

        self.title = QtWidgets.QLabel("Patient X | Admit Date: X")
        # self._beat_matching_ax.set_title(
        #     f"Patient: {self.pid}   |    "
        #     f"Admit Date: {self.date.day}-{self.date.month}-{self.date.year}\n"
        #     f"Beat Matching"
        # )
        main_layout.addWidget(self.title)

        # Create tabs for PAT and Raw data
        tab = QtWidgets.QTabWidget()

        main_layout.addWidget(tab)

        pat_page = QtWidgets.QWidget()
        self.pat_layout = QtWidgets.QGridLayout()
        pat_page.setLayout(self.pat_layout)

        raw_page = QtWidgets.QWidget()
        self.raw_page_layout = QtWidgets.QGridLayout()
        raw_page.setLayout(self.raw_page_layout)

        tab.addTab(pat_page, "PAT")
        tab.addTab(raw_page, "Raw Data")

        self._setup_pat_ui(self.pat_layout)
        self._setup_raw_ui(self.raw_page_layout)

    # Make this another class?
    def _setup_pat_ui(self, layout):
        """
        Setup the UI for the PAT data with plots and buttons etc.
        given a layout page for a tab
        """

        canvas = [FigureCanvas(Figure(figsize=(5, 3))) for _ in range(6)]

        ### Row 0 - Beat Matching display
        layout.addWidget(NavigationToolbar(canvas[0], self), 0, 0, 1, 6)
        layout.addWidget(canvas[0], 1, 0, 1, 6)

        # Dropdowns for device and patient selection
        vbutton_layout = QtWidgets.QVBoxLayout()
        layout.addLayout(vbutton_layout, 1, 6, 1, 2)
        self.year_dropdown = QtWidgets.QComboBox()
        self.year_dropdown.setFixedSize(80, 30)
        self.device_dropdown = QtWidgets.QComboBox()
        self.device_dropdown.setFixedSize(80, 30)
        self.patient_dropdown = QtWidgets.QComboBox()
        self.patient_dropdown.setFixedSize(80, 30)
        vbutton_layout.addWidget(self.year_dropdown)
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

        # Buttons to step through dynamic plots and change window params
        gbutton_layout = QtWidgets.QGridLayout()
        layout.addLayout(gbutton_layout, 5, 6, 1, 2)

        self.step_button = QtWidgets.QPushButton("Step")
        self.step_button.setFixedSize(50, 30)
        gbutton_layout.addWidget(self.step_button, 0, 0)

        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.setFixedSize(50, 30)
        gbutton_layout.addWidget(self.reset_button, 0, 1)

        # Text input for window size
        self.w_size = QtWidgets.QLineEdit()
        self.w_size.setFixedSize(40, 30)
        gbutton_layout.addWidget(self.w_size, 0, 2)

        # Text input for ST params for both sawtooths
        p1b = QtWidgets.QPushButton("Period")
        p1b.setFixedSize(50, 30)
        gbutton_layout.addWidget(p1b, 1, 0)

        self.p1 = QtWidgets.QLineEdit()
        self.p1.setFixedSize(50, 30)
        gbutton_layout.addWidget(self.p1, 1, 1)

        self.p2 = QtWidgets.QLineEdit()
        self.p2.setFixedSize(50, 30)
        gbutton_layout.addWidget(self.p2, 1, 2)

        ph1b = QtWidgets.QPushButton("Phase")
        ph1b.setFixedSize(50, 30)
        gbutton_layout.addWidget(ph1b, 2, 0)

        self.ph1 = QtWidgets.QLineEdit()
        self.ph1.setFixedSize(50, 30)
        gbutton_layout.addWidget(self.ph1, 2, 1)

        self.ph2 = QtWidgets.QLineEdit()
        self.ph2.setFixedSize(50, 30)
        gbutton_layout.addWidget(self.ph2, 2, 2)

        a1b = QtWidgets.QPushButton("Amp")
        a1b.setFixedSize(50, 30)
        gbutton_layout.addWidget(a1b, 3, 0)

        self.a1 = QtWidgets.QLineEdit()
        self.a1.setFixedSize(50, 30)
        gbutton_layout.addWidget(self.a1, 3, 1)

        self.a2 = QtWidgets.QLineEdit()
        self.a2.setFixedSize(50, 30)
        gbutton_layout.addWidget(self.a2, 3, 2)

        self.update_button = QtWidgets.QPushButton("Update Plots")
        self.update_button.setFixedSize(100, 30)
        gbutton_layout.addWidget(self.update_button, 4, 1)

        ### Row 3 - Corrected PAT Series
        # layout.addWidget(NavigationToolbar(canvas[5], self), 7, 0, 1, 2)
        # layout.addWidget(canvas[5], 8, 0, 1, 6)

        # Create subplots for each canvas
        self._beat_matching_ax = canvas[0].figure.subplots()
        self._pat_series_ax = canvas[1].figure.subplots()
        self._sawtooth_ax1 = canvas[2].figure.subplots()
        self._sawtooth_ax2 = canvas[3].figure.subplots()
        self._corrected_ax = canvas[4].figure.subplots()
        # self._final_ax = canvas[5].figure.subplots()

    def set_button_callbacks(self, step_cb, reset_cb, window_cb, st_cb, update_cb):

        self.step_button.clicked.connect(step_cb)
        self.reset_button.clicked.connect(reset_cb)
        self.w_size.textChanged.connect(window_cb)
        self.update_button.clicked.connect(update_cb)

        self.a1.textChanged.connect(lambda x: st_cb(x, 0, 0))
        self.a2.textChanged.connect(lambda x: st_cb(x, 0, 1))

        self.p1.textChanged.connect(lambda x: st_cb(x, 1, 0))
        self.p2.textChanged.connect(lambda x: st_cb(x, 1, 1))

        self.ph1.textChanged.connect(lambda x: st_cb(x, 3, 0))
        self.ph2.textChanged.connect(lambda x: st_cb(x, 3, 1))

    def _year_change(self, year_id):
        """
        Callback to update year data
        """
        self.year = int(self.year_dropdown.currentText())
        self.year_cb(self.year)

    def _device_change(self, device_id):
        """
        Callback to update device data
        """
        self.dev = int(self.device_dropdown.currentText())
        self.device_cb(self.dev)

    def _patient_change(self, patient_id):
        """
        Callback to update patient data
        """
        self.pid = int(self.patient_dropdown.currentText())
        self.patient_cb(self.pid)

    def update_button_text(self, fitp1, fitp2, window_s):

        self.a1.setText(f"{fitp1[0]:.4f}")
        self.p1.setText(f"{fitp1[1]:.2f}")
        self.ph1.setText(f"{fitp1[3]:.2f}")

        self.a2.setText(f"{fitp2[0]:.4f}")
        self.p2.setText(f"{fitp2[1]:.2f}")
        self.ph2.setText(f"{fitp2[3]:.2f}")

        self.w_size.setText(str(window_s))

    def set_title(self, title):
        self.title.setText(title)

    def add_years(self, years, callback, select_year=None):
        """
        Add years to the dropdown and set the callback for when a new
        year is selected.
        This only needs to be called once on startup.
        """
        self.year_dropdown.clear()

        for y in sorted(years):
            self.year_dropdown.addItem(f"{int(y)}")

        self.year_cb = callback
        self.year_dropdown.activated.connect(self._year_change)

        if select_year:
            self.year_dropdown.setCurrentText(str(select_year))

    def add_devices(self, devices, callback, select_device=None):
        """
        Add devices to the dropdown and set the callback for when a new
        device is selected.
        This only needs to be called once on startup.
        """
        self.device_dropdown.clear()

        for d in sorted(devices):
            self.device_dropdown.addItem(f"{int(d)}")

        self.device_cb = callback
        self.device_dropdown.activated.connect(self._device_change)

        if select_device:
            self.device_dropdown.setCurrentText(str(select_device))

        callback(self.device_dropdown.currentText())

    def add_patients(self, patients, callback, select_patient=None):
        """
        Add patients to the dropdown and set the callback for when a new
        patient is selected.
        This can be called at any time to refresh patient list when a new device
        is selected.
        """
        self.patient_dropdown.clear()

        for p in sorted(patients):
            self.patient_dropdown.addItem(f"{int(p)}")

        self.patient_cb = callback
        self.patient_dropdown.activated.connect(self._patient_change)

        if select_patient:
            self.patient_dropdown.setCurrentText(str(select_patient))

    def _setup_raw_ui(self, layout):
        """
        Setup the UI for the raw data with plots and buttons etc.
        """

        ppg_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        ecg_canvas = FigureCanvas(Figure(figsize=(5, 3)))

        layout.addWidget(NavigationToolbar(ppg_canvas, self), 0, 0)
        layout.addWidget(ppg_canvas, 1, 0)

        layout.addWidget(NavigationToolbar(ecg_canvas, self), 2, 0)
        layout.addWidget(ecg_canvas, 3, 0)

        self._ppg_ax = ppg_canvas.figure.subplots()
        self._ecg_ax = ecg_canvas.figure.subplots()

    def plot_raw_data(self, ppg, ecg):
        """
        Add raw data to the plots
        """

        ppg = {"times": ppg, "values": np.full_like(ppg, 1)}
        ecg = {"times": ecg, "values": np.full_like(ecg, 1)}

        self._ppg_ax.clear()
        self._ppg_ax.scatter(ppg["times"], ppg["values"])
        self._ppg_ax.figure.canvas.draw()

        self._ecg_ax.clear()
        self._ecg_ax.scatter(ecg["times"], ecg["values"])
        self._ecg_ax.figure.canvas.draw()

        ecg_poincare = poincare(
            rpeaks=ecg["times"],
            figsize=(5, 5),
            show=False,
            ellipse=False,
            vectors=True,
            legend=True,
        )
        ppg_poincare = poincare(
            rpeaks=ppg["times"],
            figsize=(5, 5),
            show=False,
            ellipse=False,
            vectors=True,
            legend=True,
        )

        ppg_poincare = FigureCanvas(ppg_poincare[0])
        ecg_poincare = FigureCanvas(ecg_poincare[0])

        self.raw_page_layout.addWidget(NavigationToolbar(ppg_poincare, self), 0, 1)
        self.raw_page_layout.addWidget(ppg_poincare, 1, 1)
        self.raw_page_layout.addWidget(NavigationToolbar(ecg_poincare, self), 2, 1)
        self.raw_page_layout.addWidget(ecg_poincare, 3, 1)

        self._ppg_ax.set_xlabel("Time (s)")
        self._ppg_ax.set_ylabel("Value")
        self._ppg_ax.set_title("PPG")
        self._ppg_ax.grid(True)
        self._ppg_ax.figure.tight_layout()
        self._ppg_ax.figure.canvas.draw()

        self._ecg_ax.set_xlabel("Time (s)")
        self._ecg_ax.set_ylabel("Value")
        self._ecg_ax.set_title("ECG")
        self._ecg_ax.grid(True)
        self._ecg_ax.figure.tight_layout()
        self._ecg_ax.figure.canvas.draw()

        self._ppg_ax.sharex(self._ecg_ax)

    def plot_beat_match_data(self, df):
        """
        Add beat matching data to the plot
        """

        self._beat_matching_ax.clear()

        for c in df.columns:
            self._beat_matching_ax.plot(df.index, df[c], ".", markersize=1.0)

        self._beat_matching_ax.set_xlabel("Time (s)")
        self._beat_matching_ax.set_ylabel("Value")
        self._beat_matching_ax.set_title("Beat Matching Possibilities")
        self._beat_matching_ax.grid(True)
        self._beat_matching_ax.legend(df.columns, loc="upper right")
        self._beat_matching_ax.figure.tight_layout()
        self._beat_matching_ax.figure.canvas.draw()

        self._beat_matching_ax.sharex(self._pat_series_ax)

    def plot_pat_data(self, pats, c_pats, cmap=None):
        """
        Add PAT values to the plot
        """

        self._pat_series_ax.clear()

        self._pat_series_ax.scatter(
            pats["times"], pats["values"], s=3, c=cmap, cmap="Reds", alpha=0.9
        )

        self._pat_series_ax.scatter(
            c_pats["times"], c_pats["values"], s=0.3, c="blue", alpha=0.3
        )

        # median = np.median(pats["values"])
        # yrange = 0.5
        # self.ymin = median - yrange
        # self.ymax = median + yrange
        # self._pat_series_ax.set_ylim(self.ymin, self.ymax)

        self._pat_series_ax.set_xlabel("Time (s)")
        self._pat_series_ax.set_ylabel("Value")
        self._pat_series_ax.set_title("PATs with corrected outliers")
        self._pat_series_ax.grid(True)
        self._pat_series_ax.figure.tight_layout()
        self._pat_series_ax.figure.canvas.draw()

        self._pat_series_ax.sharex(self._beat_matching_ax)

    def plot_sawtooth_one(self, x, y, st_x, st_y, fitp):

        self._sawtooth_ax1.clear()

        self._sawtooth_ax1.set_xlabel("Time (s)")
        self._sawtooth_ax1.set_ylabel("PAT (s)")
        self._sawtooth_ax1.set_title(f"First sawtooth fit")
        self._sawtooth_ax1.grid(True)

        self._a1_p1 = self._sawtooth_ax1.plot(x, y, ".")
        self._a1_p2 = self._sawtooth_ax1.plot(st_x, st_y, "--", color="red")[0]

        self._sawtooth_ax1.set_xlim(np.min(x), np.max(x))
        self._sawtooth_ax1.set_ylim(np.median(y) - 0.05, np.median(y) + 0.05)

        self._sawtooth_ax1.figure.tight_layout()
        self._sawtooth_ax1.figure.canvas.draw()

    def plot_sawtooth_two(self, x, y, st_x, st_y, fitp):

        self._sawtooth_ax2.clear()

        self._sawtooth_ax2.set_xlabel("Time (s)")
        self._sawtooth_ax2.set_ylabel("PAT (s)")
        self._sawtooth_ax2.set_title(f"Second sawtooth fit")
        self._sawtooth_ax2.grid(True)

        self._a2_p1 = self._sawtooth_ax2.plot(x, y, ".")
        self._a2_p2 = self._sawtooth_ax2.plot(st_x, st_y, "--", color="red")[0]

        self._sawtooth_ax2.sharex(self._sawtooth_ax1)
        self._sawtooth_ax2.sharey(self._sawtooth_ax1)

        self._sawtooth_ax2.figure.tight_layout()
        self._sawtooth_ax2.figure.canvas.draw()

    def plot_corrected(self, x, y):

        self._corrected_ax.clear()

        self._corrected_ax.set_xlabel("Time (s)")
        self._corrected_ax.set_ylabel("PAT (s)")
        self._corrected_ax.set_title(f"Corrected Window")
        self._corrected_ax.grid(True)

        self._a3_p1 = self._corrected_ax.plot(x, y, ".")

        self._corrected_ax.sharex(self._sawtooth_ax1)
        self._corrected_ax.sharey(self._sawtooth_ax1)

        self._corrected_ax.figure.tight_layout()
        self._corrected_ax.figure.canvas.draw()
