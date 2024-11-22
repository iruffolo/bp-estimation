import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.optimize import curve_fit, differential_evolution


def create_sawtooth(x, A, period, offset, phase):
    """
    Create a sawtooth wave

    :param x: x values
    :param A: Amplitude (ms)
    :param freq: Period (s)
    :param offset: Offset (s)
    :param phase: Phase shift (radians)
    """

    x_scaled = x / x[-1]
    freq_scaled = (np.max(x) - np.min(x)) / period

    A_sec = A / 1000

    st = A_sec * signal.sawtooth((2 * np.pi * freq_scaled) * x_scaled - phase) + offset

    return st


def fit_sawtooth(x, y, period=60, amp=20):
    """
    Fit a sawtooth wave to the data

    :param x: x values
    :param y: y values
    """

    offset = np.median(y)

    lower = [10, period - 20, offset - 0.02, 0]
    upper = [30, period + 20, offset + 0.02, 2 * np.pi]

    bounds = (lower, upper)

    io = [amp, period, offset, 0]

    fitP, pcov = curve_fit(create_sawtooth, x, y, p0=io, bounds=bounds)

    model = create_sawtooth(x, *fitP)

    return model, fitP


def sawtooth_error(y, st):

    error = st - y
    mse = np.mean(np.square(error))
    rmse = np.sqrt(mse)

    # print(f"RMSE: {rmse}")
    return rmse


if __name__ == "__main__":
    x = np.load("x2.npy")
    y = np.load("y2.npy")

    x = x - x[0]
    plt.plot(x, y, "b.")

    x_scaled = x / x[-1]
    x_linspace = np.linspace(min(x), max(x), num=1000)
    x_st = x_linspace / x_linspace[-1]

    x_range = np.max(x) - np.min(x)

    a = (np.median(y) + np.std(y)) - (np.median(y) - np.std(y))
    period = x_range / 60
    offset = np.median(y)
    phase = 1 * np.pi

    ip = [a, period, offset, None]
    print(f"Initial Parameters: {ip}")

    lscale = 0.8
    uscale = 1.2
    lower = [a * lscale, period * lscale, offset * lscale, 0]
    upper = [a * uscale, period * uscale, offset * uscale, 2 * np.pi]
    bounds = (lower, upper)
    fitP, pcov = curve_fit(create_sawtooth, x_scaled, y, bounds=bounds)

    print(f"Fitted parameters: {fitP}")

    y_st1 = create_sawtooth(x_scaled, *fitP)
    y_st2 = create_sawtooth(x_scaled, a, period, offset, phase)
    e1 = sawtooth_error(y, y_st1)
    e2 = sawtooth_error(y, y_st2)
    print(f"Error 1: {e1}")
    print(f"Error 2: {e2}")

    phase_shifts = np.linspace(0, np.pi, num=50)

    best = fitP[3]
    for ps in phase_shifts:
        y_st = create_sawtooth(x_scaled, a, period, offset, ps)
        e = sawtooth_error(y, y_st)

        if e < e1:
            best = ps
        print(f"Phase: {ps}, Error: {e}")

    print(f"Best phase: {best}")
    print(f"Old phase: {fitP[3]}")

    y_st1 = create_sawtooth(x_st, *fitP)
    y_st2 = create_sawtooth(x_st, fitP[0], fitP[1], fitP[2], best)
    plt.plot(x_linspace, y_st1, "--", color="green")
    plt.plot(x_linspace, y_st2, "--", color="red")

    plt.show()

    # fitP, pcov = curve_fit(create_sawtooth, x_scale, y, ip)
    # fitP, pcov = curve_fit(
    #     create_sawtooth,
    #     x_scale,
    #     y,
    #     ip,
    #     bounds=([0.02, 0, 0.5, 0], [0.05, 500, 2.5, 2 * np.pi]),
    #     method="dogbox",
    # )
    # print(f"Fitted parameters: {fitP}")
    # model = create_sawtooth(x_scale, *fitP)
    # x_st = np.linspace(min(x), max(x), num=200)
    # y_st = create_sawtooth(x_st, *fitP)
    # plt.plot(x_st, y_st, "--")

    # # Random experiment - Scale period with X length
    # points = 5000
    # x = np.linspace(0, 1000, points)
    # period = 5
    # phase = 0 * np.pi
    # x_scale = x / x[-1]
    # plt.plot(x, scipy.signal.sawtooth(2 * np.pi * period * x_scale - phase))
    # plt.show()
