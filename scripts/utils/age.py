import numpy as np


def calc_age(birthday, times):
    """
    Calculate age of patient

    :param birthday: Birthday of patient
    :param times: Times of signal

    :return: Age of patient
    """

    age = times - birthday

    return age


if __name__ == "__main__":
    bday = np.datetime64("1990-01-01")
    t = [np.datetime64("2020-01-01"), np.datetime64("2020-01-02")]

    print(calc_age(bday, t))
