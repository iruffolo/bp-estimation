from typing import Literal

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize, LinearConstraint
import scipy.sparse


def piecewise(
    y: np.ndarray, n: int, init: Literal["2diff", "dist"] = "2diff"
) -> np.ndarray:
    """
    :param y: Length-m array to approximate
    :param n: Number of segments
    :param init: Initialization method, either 2nd-order differential or evenly distributed
    :return: (n+1) segment bound indices; will always include 0 and m-1
    """
    m = y.size
    x_inferred = np.arange(m)

    def cost(x: np.ndarray) -> float:
        xshort = (0, *x, m - 1)
        yshort = np.interp(x=xshort, xp=x_inferred, fp=y)
        y_long = np.interp(x=x_inferred, xp=xshort, fp=yshort)
        error = y_long - y
        return error.dot(error)

    if init == "2diff":
        diff2 = np.abs(np.diff(np.diff(y)))
        x0 = 1 + np.sort(diff2.argsort()[1 - n :])
    else:
        x0 = np.linspace(start=m / n, stop=m * (n - 1) / n, num=n - 1)

    bounds = np.stack(
        (
            np.arange(1, n),
            np.arange(m - n, m - 1),
        ),
        axis=1,
    )
    increasing = LinearConstraint(
        A=scipy.sparse.eye(n - 2, n - 1, k=1) - scipy.sparse.eye(n - 2, n - 1),
        lb=1,
    )
    res = minimize(
        fun=cost,
        x0=x0,
        bounds=bounds,
        constraints=increasing,
    )
    assert res.success, res.message
    return np.concatenate(((0,), res.x.round().astype(int), (m - 1,)))


def test() -> None:
    shock = np.array(
        (
            131.7592766,
            -5.28111954,
            -5.30412333,
            6.19553924,
            -5.97658804,
            -7.83259865,
            -9.50784211,
            -15.73856643,
            -23.31825084,
            -29.48978404,
            -31.46755173,
            -33.47232039,
            -34.66509473,
            -35.74717242,
            -36.47997764,
            -37.32640433,
            -37.41553313,
            -37.81559914,
            -38.75508336,
            -38.36080882,
            -36.72118142,
            -35.74776154,
            -34.14582487,
            -32.82878478,
            -31.40182366,
            -29.97427545,
            -28.61938541,
            -24.90985539,
            -21.32175733,
            -18.73506067,
            -16.07995167,
            -16.15493052,
            -16.1433169,
        )
    )

    xdist = piecewise(y=shock, n=8, init="dist")
    x2diff = piecewise(y=shock, n=8, init="2diff")
    fig, ax = plt.subplots()
    ax.scatter(np.arange(shock.size), shock, label="original")
    ax.plot(xdist, shock[xdist], label="approx (dist)")
    ax.plot(x2diff, shock[x2diff], label="approx (2diff)")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    test()
