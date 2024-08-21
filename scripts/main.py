import concurrent.futures

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from atriumdb import AtriumSDK, DatasetDefinition
from correlation import create_aligned_data
from data_quality import DataValidator
from numpy.polynomial import Polynomial
from pat import calclulate_pat
from plotting import plot_pat, plot_pat_hist, plot_waveforms
from sawtooth import fit_sawtooth
from scipy.stats import pearsonr, spearmanr
from sklearn import linear_model
from tqdm import tqdm
from utils.atriumdb_helpers import make_device_itr, print_all_measures


def process_pat(sdk, dev, itr):
    """
    Processing function for dataset iterator

    :param sdk: AtriumSDK instance
    :param dev: Device ID
    :param itr: Dataset iterator

    :return: Dictionary of pulse arrival times for each patient in device
    """

    # Progress bar
    pbar = tqdm(total=itr._length, desc="Processing Windows")

    run_stats = {
        "successful": 0,
        "unexpected_failed": 0,
        "failed_alignment": 0,
        "poor_ecg_quality": 0,
        "poor_ppg_quality": 0,
    }
    # pat_results = {}
    window_results = []

    for i, w in enumerate(itr):

        if not w.patient_id:
            print("No patient ID")
            continue

        # # Check if patient exists in results
        # if w.patient_id not in pat_results.keys():
        #     pat_results[w.patient_id] = {
        #         "dob": pd.to_datetime(sdk.get_patient_info(w.patient_id)["dob"]),
        #         "times": np.array([]),
        #         "pat": np.array([]),
        #         "naive_pat": np.array([]),
        #     }

        # Extract specific signals and convert timescale
        ecg, ecg_freq = [(v, k[1]) for k, v in w.signals.items() if "ECG" in k[0]][0]
        ppg, ppg_freq = [(v, k[1]) for k, v in w.signals.items() if "PULS" in k[0]][0]
        abp, abp_freq = [
            (v, k[1]) for k, v in w.signals.items() if "MDC_PRESS_BLD_ART_ABP" == k[0]
        ][0]
        sbp = [v for k, v in w.signals.items() if "SYS" in k[0]][0]
        hr = [v for k, v in w.signals.items() if "PULS_RATE" in k[0]][0]

        # Convert to seconds
        ecg["times"] = ecg["times"] / 10**9
        ppg["times"] = ppg["times"] / 10**9
        abp["times"] = abp["times"] / 10**9
        sbp["times"] = sbp["times"] / 10**9

        try:
            pats, naive_pats, n_corrected = calclulate_pat(ecg, ecg_freq, ppg, ppg_freq)

            # Window has very sparse measurements, poor quality
            if pats["times"].size < 500:
                run_stats["insufficient_pats"] += 1
                continue

            # Get Sawtooth and correct PATs
            st, params = fit_sawtooth(pats["times"], pats["values"], plot=False)
            corrected_pat = pats["values"] - st + params[2]

            # Align pats with SBP values
            synced = create_aligned_data(
                corrected_pat, naive_pats["values"], pats["times"], sbp
            )

            if synced["times"].size < 300:
                run_stats["failed_alignment"] += 1
                continue

            # Get Correlation with SBP
            p1 = pearsonr(synced["pats"], synced["bp"])
            p2 = pearsonr(synced["naive_pats"], synced["bp"])
            s1 = spearmanr(synced["pats"], synced["bp"])
            s2 = spearmanr(synced["naive_pats"], synced["bp"])

            r1 = linear_model.RANSACRegressor()
            r1.fit(synced["pats"].reshape(-1, 1), synced["bp"])
            y1 = r1.predict(synced["pats"].reshape(-1, 1))

            r2 = linear_model.RANSACRegressor()
            r2.fit(synced["naive_pats"].reshape(-1, 1), synced["bp"])
            y2 = r2.predict(synced["naive_pats"].reshape(-1, 1))

            window_results.append(
                {
                    "patient_id": w.patient_id,
                    "start_time": ecg["times"][0],
                    "dob": pd.to_datetime(sdk.get_patient_info(w.patient_id)["dob"]),
                    "num_pats": synced["pats"].size,
                    "num_corrected": n_corrected,
                    "std_pats": np.std(synced["pats"]),
                    "mean_pats": np.mean(synced["pats"]),
                    "slope": r1.estimator_.coef_[0],
                    "intercept": r1.estimator_.intercept_,
                    "pearson": p1.statistic,
                    "spearman": s1.correlation,
                    "naive_std_pats": np.std(synced["naive_pats"]),
                    "naive_mean_pats": np.mean(synced["naive_pats"]),
                    "naive_slope": r2.estimator_.coef_[0],
                    "naive_intercept": r2.estimator_.intercept_,
                    "naive_pearson": p2.statistic,
                    "naive_spearman": s2.correlation,
                    "max_sbp": np.max(synced["bp"]),
                    "min_sbp": np.min(synced["bp"]),
                    "std_sbp": np.std(synced["bp"]),
                    "mean_sbp": np.mean(synced["bp"]),
                    "median_sbp": np.median(synced["bp"]),
                    "max_hr": np.max(hr["values"]),
                    "min_hr": np.min(hr["values"]),
                    "std_hr": np.std(hr["values"]),
                    "mean_hr": np.mean(hr["values"]),
                    "median_hr": np.median(hr["values"]),
                }
            )
            print(window_results[-1])
            run_stats["successful"] += 1

            # fig, ax = plt.subplots(2, figsize=(15, 10))
            # ax[0].plot(synced["pats"], synced["bp"], ".")
            # ax[0].plot(synced["pats"], y1)
            # ax[0].set_title(
            #     f"Corrected Pats ({r1.estimator_.intercept_} {r1.estimator_.coef_[0]}x)"
            # )
            # ax[0].set_xlim(0, 2)
            # ax[1].plot(synced["naive_pats"], synced["bp"], ".")
            # ax[1].plot(synced["naive_pats"], y2)
            # ax[1].set_title(
            #     f"Naive Pats ({r2.estimator_.intercept_} {r2.estimator_.coef_[0]}x)"
            # )
            # ax[1].set_xlim(0, 2)
            # plt.tight_layout()
            # plt.show()
            # plt.savefig(f"plots/corr/{w.device_id}_{w.patient_id}")

        # Peak detection faliled to detect enough peaks in calculate_pat
        except AssertionError as e:
            print(f"Signal quality issue: {e}")
            if "ECG" in str(e):
                run_stats["poor_ecg_quality"] += 1
            if "PPG" in str(e):
                run_stats["poor_ppg_quality"] += 1

        except Exception as e:
            print("Unexpected failure")
            print(e)
            run_stats["unexpected_failed"] += 1

            # results[w.patient_id]["total_corrected"] += corrected
            # results[w.patient_id]["times"] = np.concatenate(
            #     (results[w.patient_id]["times"], pats[:, 0])
            # )
            # results[w.patient_id]["pat"] = np.concatenate(
            #     (results[w.patient_id]["pat"], pats[:, 1])
            # )
            # results[w.patient_id]["naive_pat"] = np.concatenate(
            #     (results[w.patient_id]["naive_pat"], naive_pats)
            # )

            # results[w.patient_id]["successful_windows"] += 1
            # f1 = Polynomial.fit(synced["pats"], synced["bp"], 1)
            # f2 = Polynomial.fit(synced["naive_pats"], synced["bp"], 1)
            # xx, yy = f1.linspace()

            # fig, ax = plt.subplots(2, figsize=(15, 10))
            # ax[0].plot(synced["pats"], synced["bp"], ".")
            # ax[0].plot(synced["pats"], y1)
            # ax[0].set_title(f"Corrected Pats ({f1})")
            # ax[1].plot(synced["naive_pats"], synced["bp"], ".")
            # ax[1].plot(synced["naive_pats"], y2)
            # ax[1].set_title(f"Naive Pats ({f2})")
            # plt.tight_layout()
            # plt.show()
            # plt.savefig(f"plots/corr/{w.device_id}_{w.patient_id}")

            # fig, ax = plot_waveforms(ecg, ppg, abp, pats["times"], corrected_pat)
            # ax[2].plot(sbp["times"], sbp["values"])
            # ax[3].plot(naive_pats["times"], naive_pats["values"], ".")
            # plt.tight_layout()
            # plt.show()

        pbar.update(1)

        # if i > 50:
        #     break

    # if file does not exist write header
    if not os.path.isfile("filename.csv"):
        df.to_csv("filename.csv", header="column_names")
    else:  # else it exists so append without writing the header
        df.to_csv("filename.csv", mode="a", header=False)

    df = pd.DataFrame(window_results)
    df.to_csv(f"../data/results/slopes/{dev}.csv", index=False)
    # np.save(f"../data/results/device{dev}_pats.npy", results)

    np.save(f"../data/results/slopes/{dev}_runstats.npy", run_stats)

    print(f"Finished processing device {dev}")


def run(local_dataset, window_size, gap_tol, measures, device):
    """
    Function to run in parallel
    """
    sdk = AtriumSDK(dataset_location=local_dataset)
    itr = make_device_itr(sdk, device, window_size, gap_tol, measures)
    process_pat(sdk, device, itr)

    return True


if __name__ == "__main__":

    # local_dataset = "/mnt/datasets/atriumdb_abp_estimation_2024_02_05"
    # local_dataset = "/mnt/datasets/ians_data_2024_06_12"

    # Newest dataset with Philips measures (SBP, DBP, MAP) (incomplete, 90%)
    local_dataset = "/mnt/datasets/ian_dataset_2024_07_22"

    # New dataset - not tested
    # local_dataset = "/mnt/datasets/ian_dataset_2024_08_14"

    sdk = AtriumSDK(dataset_location=local_dataset)
    print_all_measures(sdk)

    devices = list(sdk.get_all_devices().keys())
    print(f"Devices: {devices}")

    # Pleth, ECG, ABP, PULSE RATE, SYS
    measures = [2, 3, 4, 7, 15]
    gap_tol = 5 * (10**9)  # 5s
    window_size = 60 * 30 * (10**9)  # 30 min

    itr = make_device_itr(sdk, 80, window_size, gap_tol, measures)
    process_pat(sdk, 80, itr)
    exit()

    num_cores = 20  # len(devices)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as pp:

        futures = {
            pp.submit(run, local_dataset, window_size, gap_tol, measures, d): d
            for d in devices
        }

        for f in concurrent.futures.as_completed(futures):
            print(f.result())

    print("Finished processing")
