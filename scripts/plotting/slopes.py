import matplotlib.pyplot as plt
import numpy as np


def plot_slopes(synced, medians, r1):

    fig, ax = plt.subplots(2, figsize=(15, 10))
    ax[0].plot(synced["pats"], synced["bp"], ".", alpha=0.5)
    ax[0].plot(medians["pats"], medians["bp"], "ro", markersize=6, label="Medians")
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
