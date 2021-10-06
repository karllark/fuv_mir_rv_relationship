import matplotlib.pyplot as plt

from astropy.table import QTable

if __name__ == '__main__':

    itab = QTable.read("results/gor09_fuse_irv_params.fits")

    gvals = itab["npts"] > 0
    plt.plot(itab["waves"][gvals], itab["slopes"][gvals])
    plt.show()
