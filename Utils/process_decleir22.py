import glob
import numpy as np
import astropy.units as u
# import matplotlib.pyplot as plt

from measure_extinction.extdata import ExtData

from rebin_extdata import rebin_extdata

if __name__ == "__main__":

    fpath = "data/decleir22/"

    files = glob.glob(f"{fpath}*.fits")

    for fname in files:
        ifile = fname
        ext = ExtData(ifile)

        next = rebin_extdata(ext, "SpeX_SXD", np.array([0.8, 2.5]) * u.micron, 500.)
        next = rebin_extdata(next, "SpeX_LXD", np.array([2.0, 5.5]) * u.micron, 500.)
        next = rebin_extdata(next, "IUE", np.array([0.100, 0.35]) * u.micron, 500.0)

        # plt.plot(ext.waves[src], ext.exts[src], "k-")
        # plt.plot(next.waves[src], next.exts[src], "r-")
        # plt.show()

        ofile = ifile.replace("decleir22/", "dec22_")
        next.save(ofile)
