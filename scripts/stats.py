import matplotlib.pyplot as plt
from datetime import datetime


def age_distribution(date_of_birth, plot=True):
    """
    Converts date of birth to an integer age.

    :param date_of_birth: Pandas df containing patient date of birth epoch time
    in nanoseconds.
    :param plot: Save a histogram plot, defaults to true.

    :return: New series of ages calculate from the date of birth
    """

    age = date_of_birth.apply(
        lambda x: (datetime.now() -
                   datetime.fromtimestamp(x/10**9)).days/365.2425).astype(int)

    if plot:
        age.hist(bins=age.nunique())
        plt.title("Patient Age Distribution")
        plt.xlabel("Age (years)")
        plt.ylabel("Count")
        plt.savefig('age_dist.png')
        plt.show()

    return age
