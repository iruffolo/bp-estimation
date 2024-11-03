import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from scipy.optimize import curve_fit


def test_func(x, a, b, phase):
    return a * np.sin(b * x - phase)


if __name__ == "__main__":
    np.random.seed(42)

    x_data = np.linspace(-5, 5, num=500)
    y_data = 2.9 * np.sin(1.5 * x_data - np.pi) + np.random.normal(size=500)

    a = 2
    b = 2
    phase = np.pi
    params, _ = curve_fit(
        test_func, x_data, y_data, bounds=([0, 0, 0], [2 * a, 2 * b, 2 * np.pi])
    )
    params1, _ = curve_fit(test_func, x_data, y_data, p0=[a, b, phase])
    params2, _ = curve_fit(test_func, x_data, y_data, p0=[a, b, 0], sigma=0.1)

    print(params)
    print(params1)
    print(params2)

    m0 = test_func(x_data, *params)
    m1 = test_func(x_data, *params1)
    m2 = test_func(x_data, *params2)

    e0 = m0 - y_data
    e1 = m1 - y_data
    e2 = m2 - y_data

    mse0 = np.mean(np.square(e0))
    mse1 = np.mean(np.square(e1))
    mse2 = np.mean(np.square(e2))
    print(f"MSE0: {mse0}")
    print(f"MSE1: {mse1}")
    print(f"MSE2: {mse2}")

    # rmse = np.sqrt(mse)
    # print(f"RMSE: {rmse}")
    # print(f"Sawtooth wave: A={fitP[0]}, fi={fitP[1]}, offset={fitP[2]}")

    plt.figure(figsize=(6, 4))
    plt.scatter(x_data, y_data, label="Data")

    plt.plot(x_data, m0, label=f"f1 (no p0)", linestyle="--")
    plt.plot(x_data, m1, label="f2 (p0 = [a, b, pi])", linestyle="-.")
    plt.plot(x_data, m2, label="f2 (p0 = [a, b, 0])", linestyle="--")

    plt.title("Fitted Sine Curve")
    plt.legend(loc="best")
    plt.show()
