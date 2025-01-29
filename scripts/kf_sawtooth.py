import matplotlib.pyplot as plt
import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter


def reset_kf(f, i0):

    f.x = np.array([i0])
    f.P *= 1000.0

    f.R = np.array([[0.01]])  # Measurement noise
    f.Q = np.array([0.000001])  # Process noise

    f.H = np.array([1.0])
    f.B = np.array([1.0])


def kf(x, y, thresh=0.02, slope=0.04 / 60, fail_count=2):

    # Expected s.t. slope = amp / period
    slope = slope

    f = KalmanFilter(dim_x=1, dim_z=1, dim_u=1)
    reset_kf(f, y[0])

    ys = []
    preds = [f.x]
    times = [x[0]]
    meas = [y[0]]

    kfs = {}
    current_kf = 0

    resids = []
    probs = []

    fails = 0
    fail_data = []

    diffs = np.diff(x[1:])

    for t, dx, m in zip(x[1:], diffs, y[1:]):

        # Check measurment before update
        y = f.residual_of(m)

        # if residual is ok, predict and update like normal
        if abs(y) < thresh:
            # if there was a previous fail, predict and update 2x
            if fails:
                for t_f, dx_f, m_f in fail_data:
                    u = dx_f * slope
                    f.predict(u=u)
                    f.update(m_f)

                    times.append(t_f)
                    meas.append(m_f)
                    preds.append(f.x)
                    ys.append(f.y)

                    fails = 0
                    fail_data = []

            u = dx * slope
            f.predict(u=u)
            f.update(m)

            times.append(t)
            meas.append(m)
            preds.append(f.x)
            ys.append(f.y)

        # if its not, do nothing
        else:
            fail_data.append((t, dx, m))
            fails += 1

            # if 2 fails, start new KF, initialize off buffer
            if fails > fail_count:
                kfs[current_kf] = {
                    "times": np.array(times).flatten(),
                    "ys": np.array(ys).flatten(),
                    "preds": np.array(preds).flatten(),
                    "meas": np.array(meas).flatten(),
                }
                current_kf += 1

                times = []
                preds = []
                ys = []
                meas = []

                reset_kf(f, fail_data[0][2])
                for t_f, dx_f, m_f in fail_data:
                    u = dx_f * slope
                    f.predict(u=u)
                    f.update(m_f)

                    times.append(t_f)
                    meas.append(m_f)
                    preds.append(f.x)
                    ys.append(f.y)

                fails = 0
                fail_data = []

        # resids.append(f.y)
        # probs.append(f.mahalanobis)

    kfs[current_kf] = {
        "times": np.array(times).flatten(),
        "ys": np.array(ys).flatten(),
        "preds": np.array(preds).flatten(),
        "meas": np.array(meas).flatten(),
    }

    return kfs


def calc_sawtooth(times, pats, fn=None, plot=False, path=None):

    # med = np.median(pats)
    # cut = np.where(abs(pats - med) < 0.02)

    x = times.values  # [cut]
    y = pats.values  # [cut]

    if plot:
        fig, ax = plt.subplots()
        ax.scatter(x, y, marker="x", label="Original Data", s=1.5)

    kfs = kf(x, y)

    params_st1 = {
        "slope_ppm": [],
        "period": [],
        "points": [],
    }
    params_st2 = {
        "slope_ppm": [],
        "period": [],
        "points": [],
    }

    data = {
        "st1": {"times": [], "values": []},
        "st2": {"times": [], "values": []},
    }

    for i, k in kfs.items():

        if len(k["times"]) < 10:
            continue

        poly = np.polyfit(k["times"], k["preds"], 1)
        poly1d = np.poly1d(poly)

        # Cut the first and last
        if i > 0 and i < len(kfs) - 1:
            slope = poly1d.c[0] * 1_000_000
            period = k["times"][-1] - k["times"][0]
            params_st1["slope_ppm"].append(slope)
            params_st1["period"].append(period)
            params_st1["points"].append(len(k["times"]))

        fixed = k["meas"] - k["preds"] + k["preds"][0]

        for i, z in enumerate(fixed):
            data["st1"]["values"].append(z)
            data["st1"]["times"].append(k["times"][i])

        if plot:
            ax.scatter(
                k["times"], k["preds"], color="green", marker=".", label="KF Predict"
            )
            ax.plot(
                k["times"],
                poly1d(k["times"]),
                color="orange",
                label="Line Fit to KF Predict",
            )
            ax.scatter(
                k["times"],
                fixed,
                color="black",
                label="Corrected by KF fit",
                s=1.5,
                alpha=0.8,
            )

    kfs2 = kf(x, data["st1"]["values"], 0.011, 0.02 / 160, 5)

    for i, k in kfs2.items():
        if len(k["times"]) < 10:
            continue

        poly = np.polyfit(k["times"], k["preds"], 1)
        poly1d = np.poly1d(poly)

        # Cut the first and last
        if i > 0 and i < len(kfs) - 1:
            slope = poly1d.c[0] * 1_000_000
            period = k["times"][-1] - k["times"][0]
            params_st2["slope_ppm"].append(slope)
            params_st2["period"].append(period)
            params_st2["points"].append(len(k["times"]))

        if plot:
            ax.scatter(
                k["times"], k["preds"], color="purple", marker=".", label="KF Predict"
            )
            ax.plot(
                k["times"],
                poly1d(k["times"]),
                color="blue",
                label="Line Fit to KF Predict",
            )

        fixed = k["meas"] - k["preds"] + k["preds"][0]

        for i, z in enumerate(fixed):
            data["st2"]["values"].append(z)
            data["st2"]["times"].append(k["times"][i])

    if plot:
        plt.legend(
            [
                "Original Data",
                "KF Predict",
                "KF Predict Line Fit",
                "Corrected by KF Fit",
            ],
            loc="upper right",
        )
        plt.tight_layout()
        # if fn:
        # plt.savefig(f"../data/st_plots_post/{fn}.png")
        plt.show()
        plt.close()

    return (
        pd.DataFrame(data["st1"]),
        pd.DataFrame(data["st2"]),
        params_st1,
        params_st2,
    )
