import numpy as np


def print_stats(data):

    print(f"Number of windows: {len(data)}")
    print(f"Number of valid windows: {sum(data.valid)}")
    print(f"Avg mean: {np.nanmean(data.mean)}")
    print(f"Avg max: {np.nanmean(data.max)}")
    print(f"Avg min: {np.nanmean(data.min)}")
    print(f"Avg std: {np.nanmean(data.std)}")


def intersection(x, y, z):

    res = np.logical_and(np.logical_and(x, y), z)

    return sum(res)


if __name__ == "__main__":

    x = np.load("stats.npy", allow_pickle=True).item()

    print("ABP")
    print_stats(x.abp)

    print()
    print("ECG")
    print_stats(x.ecg)

    print()
    print("PPG")
    print_stats(x.ppg)

    print()
    print(f"Number of windows with valid ABP/ECG/PPG :"
          f"{intersection(x.abp.valid, x.ecg.valid, x.ppg.valid)}")
