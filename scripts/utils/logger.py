import os
from enum import Enum

import numpy as np
import pandas as pd
from tqdm import tqdm


class WindowStatus(Enum):
    SUCCESS = 0
    NO_PATIENT_ID = 1
    UNEXPECTED_FAILURE = 2
    FAILED_BP_ALIGNMENT = 3
    INCOMPLETE_WINDOW = 4
    POOR_ECG_QUALITY = 5
    POOR_PPG_QUALITY = 6
    INSUFFICIENT_PATS = 7


class Logger:

    def __init__(self, dev, total_windows, path=None, verbose=False):
        """
        Initialize the logger

        :param dev: The device ID
        :param path: The path to the log file
        :param verbose: Whether to print verbose output
        """
        if path is None:
            raise ValueError("Path not specified")

        self.dev = dev
        self.path = path
        self.verbose = verbose

        if not os.path.exists(self.path):
            print(f"Creating directory {self.path}")
            os.makedirs(self.path)

        # Progress bar
        if verbose:
            self.pbar = tqdm(
                total=total_windows, desc=f"Processing Window Device {dev}"
            )

        self.total_windows = total_windows

        # Initialize error counts
        self.window_stats = {
            WindowStatus.SUCCESS.name: 0,
            WindowStatus.NO_PATIENT_ID.name: 0,
            WindowStatus.UNEXPECTED_FAILURE.name: 0,
            WindowStatus.FAILED_BP_ALIGNMENT.name: 0,
            WindowStatus.INCOMPLETE_WINDOW.name: 0,
            WindowStatus.POOR_ECG_QUALITY.name: 0,
            WindowStatus.POOR_PPG_QUALITY.name: 0,
            WindowStatus.INSUFFICIENT_PATS.name: 0,
        }

        # Every X windows, log results then reset
        self.log_rate = 10
        self.results = []

    def log_status(self, status: WindowStatus):
        """
        Log an error or success for a window

        :param status: The error to log
        """

        self.window_stats[status.name] += 1

        if self.verbose:
            print(f"Window Status: {status.name}")
            self.pbar.update(1)

    def log_data(self, pid, dob, start, n_corrected, s1, s2, hr, synced):
        """
        Log the data for a window
        """

        res = {
            "patient_id": pid,
            "dob": dob,
            "start_time": start,
            "spearman": s1.correlation,
            "naive_spearman": s2.correlation,
            "num_pats": synced["pats"].size,
            "num_corrected": n_corrected,
            "std_pats": np.std(synced["pats"]),
            "mean_pats": np.mean(synced["pats"]),
            "naive_std_pats": np.std(synced["naive_pats"]),
            "naive_mean_pats": np.mean(synced["naive_pats"]),
            "max_sbp": np.max(synced["bp"]),
            "min_sbp": np.min(synced["bp"]),
            "std_sbp": np.std(synced["bp"]),
            "mean_sbp": np.mean(synced["bp"]),
            "median_sbp": np.median(synced["bp"]),
            "max_hr": np.nanmax(hr["values"]),
            "min_hr": np.nanmin(hr["values"]),
            "std_hr": np.nanstd(hr["values"]),
            "mean_hr": np.nanmean(hr["values"]),
            "median_hr": np.nanmedian(hr["values"]),
        }

        if self.verbose:
            os.system("clear")
            print(res)

        self.results.append(res)

        # Log results every X windows and clear array
        if len(self.results) > self.log_rate:
            self._save_current_res()

    def log_raw_data(self, data, filename):
        """
        Log raw data for a window
        """

        df = pd.DataFrame(data)

        fn = os.path.join(self.path, f"device_{self.dev}_{filename}.csv")

        # if file does not exist write header, else append
        if not os.path.isfile(fn):
            df.to_csv(fn, header="column_names", index=False)
        else:
            df.to_csv(fn, mode="a", header=False, index=False)

    def _save_current_res(self):
        """
        Save the current results to a CSV file and clear the array
        """
        df = pd.DataFrame(self.results)

        fn = os.path.join(self.path, f"device_{self.dev}_summary_data.csv")

        # if file does not exist write header, else append
        if not os.path.isfile(fn):
            df.to_csv(fn, header="column_names", index=False)
        else:
            df.to_csv(fn, mode="a", header=False, index=False)

        self.results.clear()

    def print_stats(self):
        """
        Print total error counts for device
        """

        print(f"Error Counts for device {self.dev}:")
        print(f"Total Windows: {self.total_windows}")
        for status in self.window_stats:
            print(f"{status}: {self.window_stats[status]}")

    def save(self):
        """
        Save the log to a CSV file
        """

        print(self.total_windows)
        print(self.window_stats)

        df = pd.DataFrame(
            {
                "device": self.dev,
                "total_windows": self.total_windows,
                **self.window_stats,
            },
            index=[0],
        )

        df.to_csv(os.path.join(self.path, f"device_{self.dev}_log.csv"), index=False)

        # Dump any remaining data
        self._save_current_res()


if __name__ == "__main__":

    log = Logger(80, 2, ".", verbose=True)

    log.log_status(WindowStatus.INCOMPLETE_WINDOW)
    log.log_status(WindowStatus.NO_PATIENT_ID)

    log.print_stats()

    log.save()
