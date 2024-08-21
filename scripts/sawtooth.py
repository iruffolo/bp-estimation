import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.signal
from scipy.optimize import curve_fit, differential_evolution


def _create_sawtooth(x, A, freq, offset):
    """
    Create a sawtooth wave

    :param x: x values
    :param A: Amplitude
    :param freq: Frequency
    :param offset: Offset
    """
    return A * scipy.signal.sawtooth(x / freq, width=1) + offset


def fit_sawtooth(x, y, plot=False):
    """
    Fit a sawtooth wave to the data

    :param x: x values
    :param y: y values
    """

    ip = generate_Initial_Parameters(x, y)
    # print(f"Initial Parameters: {ip}")

    # ip = [0.04, 8, 1.4]
    fitP, pcov = curve_fit(
        _create_sawtooth,
        x,
        y,
        ip,
        bounds=([0.01, 1, 0.5], [0.1, 20, 3.5]),
        method="dogbox",
    )
    # print(f"Fitted parameters: {fitP}")

    model = _create_sawtooth(x, *fitP)

    # error = model - y
    # mse = np.mean(np.square(error))
    # rmse = np.sqrt(mse)
    # print(f"RMSE: {rmse}")
    # print(f"Sawtooth wave: A={fitP[0]}, fi={fitP[1]}, offset={fitP[2]}")

    if plot:
        f = plt.figure(figsize=(15, 10), dpi=100)
        axes = f.add_subplot(111)

        # first the raw data as a scatter plot
        axes.plot(x, y, "b.")

        # create data for the fitted equation plot
        xModel = np.linspace(min(x), max(x), num=200)
        yModel = _create_sawtooth(xModel, *fitP)

        # now the model as a line plot
        # axes.plot(xModel, yModel, "rx")
        axes.plot(xModel, yModel)
        axes.set_xlabel("X Data")  # X axis data label
        axes.set_ylabel("Y Data")  # Y axis data label

        plt.show()
        plt.close("all")  # clean up after using pyplot

    return model, fitP


def generate_Initial_Parameters(x, y):
    # min and max used for bounds
    parameterBounds = list()
    parameterBounds.append([0.03, max(y) - min(y)])  # search bounds for A
    parameterBounds.append([1, 20])  # search bounds for fi
    parameterBounds.append([min(y), max(y)])  # search bounds for Offset

    # function for genetic algorithm to minimize (sum of squared error)
    def sumOfSquaredError(parameterTuple):
        val = _create_sawtooth(x, *parameterTuple)
        return np.sum((y - val) ** 2.0)

    result = differential_evolution(sumOfSquaredError, parameterBounds, seed=42)
    return result.x


if __name__ == "__main__":
    xData = np.array(
        [
            -500.0,
            -400.0,
            -300.0,
            -200.0,
            -100.0,
            0.0,
            100.0,
            200.0,
            300.0,
            400.0,
            500.0,
        ]
    )
    yData = np.array(
        [-24.0, 73.0, 55.0, 36.0, 18.0, 0.0, -18.0, 79.0, 61.0, 43.0, 24.0]
    )

    fit_sawtooth(xData, yData)
