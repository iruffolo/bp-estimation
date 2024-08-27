from atriumdb import AtriumSDK, DatasetDefinition


def make_patients_dataset(sdk, meas, gap_tol_s=1,
                          pats_per_ds=100, path="datasets/patients"):
    """
    Creates multiple DatasetDefinitions by patient and saves them to folder,
    for faster loading.

    :param sdk: AtriumDB sdk.
    :param meas: List of measures to use in the Dataset.
    :param gap_tol_s: Gap tolerance (missing data) allowed, in seconds.
    :param pats_per_ds: Number of patients to include in each Dataset.
    :param path: Path to folder to save Datasets.

    :return: None
    """

    patients = sdk.get_all_patients()
    ids = {k: 'all' for k in patients.keys()}

    gap_tol_ns = gap_tol_s * (10 ** 9)

    for i in range(0, len(patients), pats_per_ds):

        p = {k: ids[k] for k in list(ids)[i:i+pats_per_ds]}

        print(f"Building dataset, patients: {i}:{i+pats_per_ds}")

        definition = DatasetDefinition.build_from_intervals(
            sdk, "measures", measures=meas,
            patient_id_list=p,
            merge_strategy="intersection",
            gap_tolerance=gap_tol_ns)

        definition.save(f"{path}/pats_{i}_{i+pats_per_ds}.yaml", force=True)


def make_devices_dataset(sdk, meas, gap_tol_s=1, path="datasets/devices"):
    """
    Creates multiple DatasetDefinitions by device and saves them to folder,
    for faster loading.

    :param sdk: AtriumDB sdk.
    :param meas: List of measures to use in the Dataset.
    :param gap_tol_s: Gap tolerance (missing data) allowed, in seconds.
    :param path: Path to folder to save Datasets.

    :return: None
    """

    devices = sdk.get_all_devices()

    gap_tol_ns = gap_tol_s * (10 ** 9)

    for dev in devices:

        print(f"Building dataset, device: {dev}")

        definition = DatasetDefinition.build_from_intervals(
            sdk, "measures", measures=meas,
            device_id_list={dev: "all"},
            merge_strategy="intersection",
            gap_tolerance=gap_tol_ns)

        definition.save(f"{path}/dev_{dev}.yaml", force=True)


if __name__ == "__main__":

    local_dataset = "/mnt/datasets/atriumdb_abp_estimation_2024_02_05"

    sdk = AtriumSDK(dataset_location=local_dataset)

    measures = [
        {'tag': "MDC_PRESS_BLD_ART_ABP",
            'freq_nhz': 125_000_000_000, 'units': "MDC_DIM_MMHG"},
        {'tag': "MDC_ECG_ELEC_POTL_II", 'freq_nhz': 500_000_000_000,
            'units': "MDC_DIM_MILLI_VOLT"},
        {'tag': "MDC_PULS_OXIM_PLETH", 'freq_nhz': 125_000_000_000,
            'units': "MDC_DIM_DIMLESS"},
    ]

    make_devices_dataset(sdk, measures)
    make_patients_dataset(sdk, measures)
