import glob
# import numpy as np

from measure_extinction.extdata import ExtData

if __name__ == "__main__":

    fpath = "data/gordon21/"

    files = glob.glob(f"{fpath}*.fits")

    for fname in files:
        ifile = fname
        ext = ExtData(ifile)
        ext.calc_RV()

        ofile = ifile.replace("gordon21/", "gor21_")
        ofile = ofile.replace("_POWLAW2DRUDE", "")
        ext.save(ofile)
