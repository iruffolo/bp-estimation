import heartpy as hp
import neurokit2 as nk
from biosppy.signals.ppg import find_onsets_kavsaoglu2016


def ppg_peak_detect(signal_times, signal_values, freq_hz):
    """
    Get peaks from PPG

    :param signal_times: PPG signal times
    :param signal_values: PPG signal values
    :param freq_hz: Frequency of signal

    :return: Timestamps of all peaks within signal_times
    """

    # signal = hp.filter_signal(signal, sample_rate=freq,
    # cutoff=40, order=2, filtertype='lowpass')
    # working_data, measures = hp.process(signal_values, sample_rate=freq_hz)

    onsets = find_onsets_kavsaoglu2016(
        signal=signal_values,
        sampling_rate=freq_hz,
        init_bpm=90,
        min_delay=0.2,
        max_BPM=247,
    )[0]

    return signal_times[onsets]


def rpeak_detect_fast(signal_times, signal_values, freq_hz):
    """
    Detect R-peaks using NeuroKit2

    :param signal_values: Values of signal
    :param freq_hz: Frequency of signal

    :return: R-peak indices
    """

    try:
        clean_signal = nk.ecg_clean(signal_values, sampling_rate=int(freq_hz))
        # Set mindelay to control max heartrate. 250bpm = 0.24
        signals, info = nk.ecg_peaks(
            clean_signal, sampling_rate=int(freq_hz), mindelay=0.24
        )

        peak_indices = info["ECG_R_Peaks"]

        # Correct Peaks
        (corrected_peak_indices,) = ecg.correct_rpeaks(
            signal=signal_values, rpeaks=peak_indices, sampling_rate=freq_hz
        )

        return signal_times[corrected_peak_indices]

    except Exception as e:
        print(f"Error: {e}")
        return np.array([])
