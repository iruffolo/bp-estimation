import numpy as np
import pandas as pd

import os
from visualization.beat_matching import (
    beat_matching,
    correct_pats,
    peak_detect,
    rpeak_detect_fast,
)

import matplotlib.pyplot as plt

if __name__=="__main__":

    path = "/home/ian/dev/bp-estimation/data/artifact/"

    files = os.listdir(path)

    ecg_files = [f for f in files if "ECG" in f]
    ppg_files = [f for f in files if "PLETH" in f]

    # ppg_peak_times = peak_detect(ppg["times"], ppg["values"], ppg_freq)
    ppg = pd.read_csv(path+ppg_files[0], names =["times", "values"])

    ppg_peak_times = peak_detect(ppg["times"], ppg["values"], 125).values

    print(ppg_peak_times)

    csum = np.cumsum(np.diff(ppg_peak_times))

    x = ppg_peak_times[:-1] - ppg_peak_times[0]

    poly = np.polyfit(x, csum, 1)
    poly1d = np.poly1d(poly)

    y = csum - poly1d(x)

    fig, ax = plt.subplots(1,3)
    ax[0].plot(ppg_peak_times[:-1], np.diff(ppg_peak_times))
    ax[1].scatter(x, csum, s=0.5, marker='x')
    ax[1].plot(x, poly1d(x))
    ax[1].set_title("cumsum")
    ax[2].scatter(x, y, s=0.5)
    ax[2].set_title("detrended cumsum")

    plt.show()



