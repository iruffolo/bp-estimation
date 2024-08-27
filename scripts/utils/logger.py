import os


def Logger():

    def __init__(path=None, verbose=False):

        self.count = 0
        self.results = []

        self.verbose = verbose

    def log(pid, dob, start, data):

        if verbose:
            print(data)


if __name__ == "__main__":

    log = Logger()
