import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyPPG.biomarkers as BM
import pyPPG.fiducials as FP
import pyPPG.ppg_sqi as SQI
import pyPPG.preproc as PP
from dotmap import DotMap
from pyPPG import PPG, Biomarkers, Fiducials
from pyPPG.datahandling import load_data, plot_fiducials, save_data


def proc_ppg(ppg_data, ppg_freq):
    """
    Process PPG data and calculate SQI

    :param ppg_data: Dict containing PPG data
    :param ppg_freq: Frequency of PPG data in Hz

    """

    # Assert no nan values in signal
    assert not np.isnan(ppg_data["values"]).any(), "Nan values in ppg signal"

    signal = DotMap()
    signal.name = "ppg_window"

    # Set signal values from input data
    signal.v = ppg_data["values"]
    signal.fs = ppg_freq

    signal.start_sig = 0  # the first sample of the signal to be analysed
    signal.end_sig = -1  # the last sample of the signal to be analysed
    signal.filtering = False  # whether or not to filter the PPG signal
    signal.fL = 0.5000001  # Lower cutoff frequency (Hz)
    signal.fH = 12  # Upper cutoff frequency (Hz)
    signal.order = 4  # Filter order
    signal.sm_wins = {
        "ppg": 50,
        "vpg": 10,
        "apg": 10,
        "jpg": 10,
    }  # smoothing windows in millisecond for the PPG, PPG', PPG", and PPG'"

    prep = PP.Preprocess(
        fL=signal.fL, fH=signal.fH, order=signal.order, sm_wins=signal.sm_wins
    )

    # Get derivative signals
    signal.ppg, signal.vpg, signal.apg, signal.jpg = prep.get_signals(s=signal)

    # Initialise the correction for fiducial points
    corr_on = ["on", "dn", "dp", "v", "w", "f"]
    correction = pd.DataFrame()
    correction.loc[0, corr_on] = True
    signal.correction = correction

    # Create a PPG class
    s = PPG(signal)

    fpex = FP.FpCollection(s=s)
    fiducials = fpex.get_fiducials(s=s)

    # Create a fiducials class
    fp = Fiducials(fp=fiducials)

    # Plot fiducial points
    plot_fiducials(s, fp, legend_fontsize=12)

    sqi = SQI.get_ppgSQI(ppg=s.ppg, fs=s.fs, annotation=fp.sp)
    print("SQI: ", sqi)

    ppgSQI = round(np.mean(sqi) * 100, 2)
    print("Mean PPG SQI: ", ppgSQI, "%")


if __name__ == "__main__":

    data = np.load("raw_data/data_0_hourly.npy", allow_pickle=True).item()
    ppg_data, ppg_freq = [
        (v, k[1] / 10**9) for k, v in data.signals.items() if "PULS" in k[0]
    ][0]

    ppg_data["values"] = ppg_data["values"][:2000]

    proc_ppg(ppg_data, ppg_freq)
