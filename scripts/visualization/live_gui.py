import sys

from atriumdb import AtriumSDK, DatasetDefinition
from gui2 import ApplicationWindow
from matplotlib.backends.qt_compat import QtWidgets
from utils.atriumdb_helpers import get_ppg_ecg_data, print_all_measures

if __name__ == "__main__":

    qapp = QtWidgets.QApplication(sys.argv)
    app = ApplicationWindow(sys.argv)

    # Mounted dataset
    # local_dataset = "/home/ian/dev/datasets/ian_dataset_2024_08_26"
    #
    # sdk = AtriumSDK(dataset_location=local_dataset)
    # print_all_measures(sdk)
    #
    # devices = list(sdk.get_all_devices().keys())
    # print(f"Devices: {devices}")
    #
    # gap_tol = 30 * 60  # 30 min to reduce overlapping windows with gap tol
    #
    # ppg, ecg = get_ppg_ecg_data(sdk, pid=31027, dev=80, gap_tol=gap_tol)
    #
    # print(f"PPG: {ppg}, ECG: {ecg}")

    app.show()
    app.activateWindow()
    app.raise_()
    qapp.exec()
