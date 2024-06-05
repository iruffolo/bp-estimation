import numpy
from atriumdb import AtriumSDK, DatasetDefinition

if __name__ == "__main__":
    print("Patient Stats")

    sdk = AtriumSDK("/mnt/datasets/atriumdb_abp_estimation_2024_02_05")

    print("Patients: ", sdk.get_all_patients())
    print("Patient 14498: ", sdk.get_patient_info(14498))

    # print(sdk.get_all_patient_encounter_data(patient_id_list=[14498]))
    all_pats = sdk.get_all_patients()

    meas_id = sdk.get_measure_id("MDC_PULS_OXIM_PLETH", freq=125_000_000_000)

    for p in all_pats:
        # print(sdk.get_patient_info(p))
        # sdk.get_device_patient_data(patient_id_list=[p])
        print("------------")

        iarray = sdk.get_interval_array(measure_id=meas_id, patient_id=p)

        # sdk.

        

        print(iarray)
