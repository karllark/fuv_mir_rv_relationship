import glob
# import numpy as np

from measure_extinction.extdata import ExtData

if __name__ == "__main__":

    fpath = "data/valencic04/"

    files = glob.glob(f"{fpath}*bin.fits")

    for fname in files:
        ifile = fname
        ext = ExtData(ifile)

        # get A(V) values
        ext.calc_AV()
        if "AV" in ext.columns.keys():
            ext.calc_RV()

            ext.type = "elx"
            ext.type_rel_band = "V"

            ofile = ifile.replace("valencic04/", "val04_")
            ext.save(ofile)
