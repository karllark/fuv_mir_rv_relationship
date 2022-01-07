import glob
import numpy as np
import astropy.units as u

from measure_extinction.extdata import ExtData

from rebin_extdata import rebin_extdata

if __name__ == "__main__":

    fpath = "data/gordon21/"

    files = glob.glob(f"{fpath}*.fits")

    for fname in files:
        ifile = fname
        ext = ExtData(ifile)
        ext.calc_RV()

        # no rebin for IRS data as resolution is below 500 already
        next = rebin_extdata(ext, "IUE", np.array([0.100, 0.35]) * u.micron, 500.0)

        ofile = ifile.replace("gordon21/", "gor21_")
        ofile = ofile.replace("_POWLAW2DRUDE", "")
        next.save(ofile)
