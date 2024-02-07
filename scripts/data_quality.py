import numpy as np
from scipy.signal import find_peaks


class DataStats:
    """
    POD dataclass
    """
    class Stats:
        def __init__(self):
            self.mean = list()
            self.std = list()
            self.var = list()
            self.max = list()
            self.min = list()
            self.num_peaks = list()
            self.pps = list()
            self.valid = list()

        def __len__(self):
            return len(self.mean)

        def __str__(self):
            return (
                f"Mean: {self.mean}\n"
                f"Std: {self.std}\n"
                f"Var: {self.var}\n"
                f"Max: {self.max}\n"
                f"Min: {self.min}\n"
                f"Num Peaks: {self.num_peaks}\n"
                f"Peaks per second: {self.pps}\n"
                f"Pass: {self.valid}\n"
            )

    def __init__(self):
        self.abp = self.Stats()
        self.ecg = self.Stats()
        self.ppg = self.Stats()

    def __str__(self):
        return (
            f"ABP: {self.abp}\n"
            f"ECG: {self.ecg}\n"
            f"PPG: {self.ppg}"
        )


class DataValidator:

    def __init__(self):

        # ABP ranges
        self.max_mean = 200
        self.min_mean = 30
        self.max_max = 300
        self.min_max = 60
        self.min_min = 20
        self.min_var = 80

        # ECG/PPG
        self.min_var_ecg = 1e-4
        self.min_peaks = 0.5  # per second
        self.max_peaks = 4.0  # per second

        self.stats = DataStats()

    def save_stats(self, data, signal):
        """
        Records the mean, std, var, max, min of signal
        """

        if signal == 'abp':
            stats = self.stats.abp
        elif signal == 'ecg':
            stats = self.stats.ecg
        elif signal == 'ppg':
            stats = self.stats.ppg
        else:
            stats = DataStats().abp

        stats.mean.append(np.mean(data))
        stats.std.append(np.std(data))
        stats.var.append(np.var(data))
        stats.max.append(np.max(data))
        stats.min.append(np.min(data))

    def valid_abp(self, abp_window):
        """
        Return true if the arterial blood pressure window passes the following
        checks:
            1) mean signal value between 30mmHg and 200mmHg
            2) max signal value between 60mmHg and 300mmHg
            3) min signal value greater than 20mmHg
            4) variance greater than 80
            5) no peaks detected with find_peaks scipy function

        Not implemented yet:
            6) diff between two consecutive peaks must be less than 50mmHg
            7) waveform is flat (doesn't change for > 2 samples)
            8) pulse pressure greater than 70mmHg

        """

        x = np.array(abp_window)
        self.save_stats(x, 'abp')

        valid = (self.min_mean <= np.mean(x) <= self.max_mean and
                 self.min_max <= np.max(x) <= self.max_max and
                 np.min(x) >= self.min_min and
                 self.min_var <= np.var(x))

        self.stats.abp.valid.append(valid)

        peaks, _ = find_peaks(x, distance=50)
        self.stats.abp.num_peaks.append(len(peaks))

        return valid, peaks

    def valid_ecg(self, ecg_window, window_size=32):
        """
        Return true if the ECG passes the following checks:
            1) variance of signal must be above a small value (1e-4)
            2) number of peaks less than 4 per second
            3) number of peaks greater than 0.5 per second
        """

        x = np.array(ecg_window)
        self.save_stats(x, 'ecg')

        peaks, _ = find_peaks(x, distance=50*4)
        self.stats.ecg.num_peaks.append(len(peaks))

        valid = (self.min_var_ecg <= np.var(x) and
                 self.min_peaks <= len(peaks)/window_size <= self.max_peaks)
        self.stats.ecg.valid.append(valid)

        return valid, peaks

    def valid_ppg(self, ppg_window, window_size=32):
        """
        Return true if the PPG passes the following checks:
            1) variance of signal must be above a small value (1e-4)
            2) number of peaks less than 4 per second
            3) number of peaks greater than 0.5 per second
        """

        x = np.array(ppg_window)
        self.save_stats(x, 'ppg')

        peaks, _ = find_peaks(x, distance=50)
        self.stats.ppg.num_peaks.append(len(peaks))

        valid = (self.min_var_ecg <= np.var(x) and
                 self.min_peaks <= len(peaks)/window_size <= self.max_peaks)
        self.stats.ppg.valid.append(valid)

        return valid, peaks

    def print_stats(self, save=False):
        """
        Prints all the saved stats with the option to save to file.
        """

        print(self.stats)

        if save:
            np.save("stats.npy", self.stats)


if __name__ == "__main__":
    print("hello stats")

    dv = DataValidator()

    x = [1, 2, 3, 4, 5]

    dv.valid_abp(x)
    dv.valid_ecg(x)
    dv.valid_ppg(x)
    dv.print_stats(save=True)
