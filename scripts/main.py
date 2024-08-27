import concurrent.futures
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from atriumdb import AtriumSDK, DatasetDefinition
from correlation import create_aligned_data
from data_quality import DataValidator
from numpy.polynomial import Polynomial
from pat import calclulate_pat
from plotting.waveforms import plot_waveforms
from sawtooth import fit_sawtooth
from scipy.stats import pearsonr, spearmanr
from sklearn import linear_model
from tqdm import tqdm
from utils.atriumdb_helpers import (make_device_itr,
                                    make_device_itr_all_signals,
                                    print_all_measures)


def process_pat(sdk, dev, itr):
    """
    Processing function for dataset iterator

    :param sdk: AtriumSDK instance
    :param dev: Device ID
    :param itr: Dataset iterator

    :return: Dictionary of pulse arrival times for each patient in device
    """

    # Progress bar
    pbar = tqdm(total=itr._length, desc=f"Processing Window Device {dev}")

    run_stats = {
        "successful": 0,
        "unexpected_failed": 0,
        "failed_alignment": 0,
        "poor_ecg_quality": 0,
        "poor_ppg_quality": 0,
        "no_patient_id": 0,
    }

    # pat_results = {}
    window_results = []

    for i, w in enumerate(itr):

        if not w.patient_id:
            run_stats["no_patient_id"] += 1
            pbar.update(1)
            continue

        # # Check if patient exists in results
        # if w.patient_id not in pat_results.keys():
        #     pat_results[w.patient_id] = {
        #         "dob": pd.to_datetime(sdk.get_patient_info(w.patient_id)["dob"]),
        #         "times": np.array([]),
        #         "pat": np.array([]),
        #         "naive_pat": np.array([]),
        #     }

        for (signal, freq, _), v in w.signals.items():
            # Skip signals without data
            if v["actual_count"] == 0:
                continue

            # Convert to s
            v["times"] = v["times"] / (10**9)

            # Extract specific signals
            match signal:
                case signal if "BEAT_RATE" in signal:
                    hr = v
                case signal if "SYS" in signal:
                    sbp = v
                case signal if "ECG_ELEC" in signal:
                    ecg, ecg_freq = v, freq
                case signal if "PULS_OXIM" in signal:
                    ppg, ppg_freq = v, freq
                case "MDC_PRESS_BLD_ART" | "MDC_PRESS_BLD_ART_ABP":
                        abp, abp_freq = v, freq
                case _:
                    pass

        try:
            pats, naive_pats, n_corrected = calclulate_pat(ecg, ecg_freq, ppg, ppg_freq)

            # Window has very sparse measurements, poor quality
            if pats["times"].size < 500:
                run_stats["insufficient_pats"] += 1
                pbar.update(1)
                continue

            # Get Sawtooth and correct PATs
            st, params = fit_sawtooth(pats["times"], pats["values"], plot=False)
            corrected_pat = pats["values"] - st + params[2]

            # Align pats with SBP values
            synced = create_aligned_data(
                corrected_pat, naive_pats["values"], pats["times"], sbp
            )

            if synced["times"].size < 300:
                print("errorrror")
                print(sbp)
                run_stats["failed_alignment"] += 1
                pbar.update(1)
                continue

            # Get Correlation with SBP
            p1 = pearsonr(synced["pats"], synced["bp"])
            p2 = pearsonr(synced["naive_pats"], synced["bp"])
            s1 = spearmanr(synced["pats"], synced["bp"])
            s2 = spearmanr(synced["naive_pats"], synced["bp"])

            # Line of best fit using RANSAC to deal with outliers
            r1 = linear_model.RANSACRegressor()
            r1.fit(synced["bp"].reshape(-1, 1), synced["pats"])

            r2 = linear_model.RANSACRegressor()
            r2.fit(synced["bp"].reshape(-1, 1), synced["naive_pats"])

            # Calculate medians for better line of best fit
            df = pd.DataFrame(synced)
            counts = df.groupby(["bp"]).size().reset_index(name="count")

            medians = df.groupby(["bp"])["pats"].median().reset_index()
            medians["count"] = counts["count"]
            naive_medians = df.groupby(["bp"])["naive_pats"].median().reset_index()
            naive_medians["count"] = counts["count"]

            # Filter by minimum number of PAT points per BP value
            medians = medians[medians["count"] > 50]
            naive_medians = naive_medians[naive_medians["count"] > 50]

            # Line of best fit using RANSAC to deal with outliers
            mr1 = linear_model.RANSACRegressor()
            mr1.fit(medians["bp"].to_numpy().reshape(-1, 1), medians["pats"])

            mr2 = linear_model.RANSACRegressor()
            mr2.fit(
                naive_medians["bp"].to_numpy().reshape(-1, 1),
                naive_medians["naive_pats"],
            )

            y1 = r1.predict(synced["bp"].reshape(-1, 1))
            y2 = r2.predict(synced["bp"].reshape(-1, 1))
            my1 = mr1.predict(synced["bp"].reshape(-1, 1))
            my2 = mr2.predict(synced["bp"].reshape(-1, 1))

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
                    "median_slope": mr1.estimator_.coef_[0],
                    "median_intercept": mr1.estimator_.intercept_,
                    "pearson": p1.statistic,
                    "spearman": s1.correlation,
                    "naive_std_pats": np.std(synced["naive_pats"]),
                    "naive_mean_pats": np.mean(synced["naive_pats"]),
                    "naive_slope": r2.estimator_.coef_[0],
                    "naive_intercept": r2.estimator_.intercept_,
                    "naive_median_slope": mr2.estimator_.coef_[0],
                    "naive_median_intercept": mr2.estimator_.intercept_,
                    "naive_pearson": p2.statistic,
                    "naive_spearman": s2.correlation,
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
            )
            print(window_results[-1])
            run_stats["successful"] += 1

            fig, ax = plt.subplots(2, figsize=(15, 10))
            ax[0].plot(synced["pats"], synced["bp"], ".", alpha=0.5)
            ax[0].plot(
                medians["pats"], medians["bp"], "ro", markersize=6, label="Medians"
            )
            ax[0].plot(
                synced["pats"],
                y1,
                label=f"Points Line ({r1.estimator_.intercept_} {r1.estimator_.coef_[0]}x)",
            )
            ax[0].plot(
                synced["pats"],
                my1,
                label=f"Medians Line ({mr1.estimator_.intercept_} {mr1.estimator_.coef_[0]}x)",
            )
            ax[0].set_title(f"Corrected Pats")
            ax[0].set_xlim(0, 2)
            ax[0].legend(loc="upper left")
            ax[0].set_xlabel("PAT (s)")
            ax[0].set_ylabel("BP (mmHG)")
            # ax[0].grid()
            ax[1].plot(synced["naive_pats"], synced["bp"], ".", alpha=0.5)
            ax[1].plot(
                naive_medians["naive_pats"],
                naive_medians["bp"],
                "ro",
                markersize=5,
                label="Medians",
            )
            ax[1].plot(
                synced["naive_pats"],
                y2,
                label=f"Points Line ({r2.estimator_.intercept_} {r2.estimator_.coef_[0]}x)",
            )
            ax[1].plot(
                synced["naive_pats"],
                my2,
                label=f"Medians Line ({mr2.estimator_.intercept_} {mr2.estimator_.coef_[0]}x)",
            )
            ax[1].set_title(
                f"Naive Pats ({r2.estimator_.intercept_} {r2.estimator_.coef_[0]}x)"
            )
            ax[1].set_xlim(0, 2)
            ax[1].legend(loc="upper right")
            ax[1].set_xlabel("PAT (s)")
            ax[1].set_ylabel("BP (mmHG)")
            # ax[1].grid()
            plt.tight_layout()
            plt.show()
            # plt.savefig(f"plots/slopes/{w.device_id}_{w.patient_id}")
            plt.close()

        # Peak detection faliled to detect enough peaks in calculate_pat
        except AssertionError as e:
            print(f"Signal quality issue: {e}")
            if "ECG" in str(e):
                run_stats["poor_ecg_quality"] += 1
            if "PPG" in str(e):
                run_stats["poor_ppg_quality"] += 1
            else:
                run_stats["unexpected_failed"] += 1

            # # Debug plot
            # plot_waveforms(ecg, ppg, abp, show=True)

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

        if len(window_results) > 50:
            df = pd.DataFrame(window_results)
            fn = f"../data/results/median_slopes/{dev}.csv"

            # if file does not exist write header, else append
            if not os.path.isfile(fn):
                df.to_csv(fn, header="column_names", index=False)
            else:
                df.to_csv(fn, mode="a", header=False, index=False)

            window_results.clear()

        pbar.update(1)

        # if i > 200:
        #     break

    # np.save(f"../data/results/device{dev}_pats.npy", results)
    np.save(f"../data/results/median_slopes/{dev}_runstats.npy", run_stats)

    print(f"Finished processing device {dev}")


def run(local_dataset, window_size, gap_tol, device):
    """
    Function to run in parallel
    """
    sdk = AtriumSDK(dataset_location=local_dataset)
    itr = make_device_itr_all_signals(sdk, device, window_size, gap_tol)
    process_pat(sdk, device, itr)

    return True


if __name__ == "__main__":

    # Newest dataset with Philips measures (SBP, DBP, MAP) (incomplete, 90%)
    # local_dataset = "/mnt/datasets/ian_dataset_2024_07_22"

    # Newest dataset
    local_dataset = "/mnt/datasets/ian_dataset_2024_08_15"

    sdk = AtriumSDK(dataset_location=local_dataset)
    print_all_measures(sdk)

    devices = list(sdk.get_all_devices().keys())
    print(f"Devices: {devices}")

    window_size = 60 * 60  # 60 min
    gap_tol = 3  # 5 min to reduce overlapping windows

    itr = make_device_itr_all_signals(sdk, 80, window_size, gap_tol, 1)
    process_pat(sdk, 80, itr)
    exit()

    num_cores = 10  # len(devices)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as pp:

        futures = {
            pp.submit(run, local_dataset, window_size, gap_tol, d): d for d in devices
        }

        for f in concurrent.futures.as_completed(futures):
            print(f.result())

    print("Finished processing")
