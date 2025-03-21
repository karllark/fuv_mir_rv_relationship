import glob
import numpy as np
# import matplotlib.pyplot as plt
from scipy.io import readsav

import astropy.units as u

from measure_extinction.extdata import ExtData as ExtDataStock
from rebin_extdata import rebin_extdata

# from dust_extinction.averages import RRP89_MWGC


class ExtData(ExtDataStock):
    """
    Expand the stock ExtData to have a function to read in Ed Fitzpatrick's
    idl save files.
    """

    def read_ext_data_idlsave(self, ext_filename):
        """
        Read the calculated extinction curve from an idl save file
        """
        self.type = "elx"
        self.type_rel_band = "V"

        spec_dict = readsav(ext_filename)

        sindxs = np.argsort(np.absolute(1.0 / spec_dict["XVALS"] - 0.44))
        self.columns["EBV"] = (
            spec_dict["E44MIN55"],
            max(spec_dict["EXTCURV_RAW_SIG"][sindxs[0]], 0.02)
        )
        print("E4455", self.columns["EBV"])

        # updated Mar 2025 to use the 4455 extinction directly
        # previously used RAW, not quite normalized at V band (off by a very small amount)
        (indxs,) = np.where(1.0 / spec_dict["XVALS"] < 1.0)
        self.waves["STIS"] = (1.0 / spec_dict["XVALS"][indxs]) * u.micron
        self.exts["STIS"] = spec_dict["EXTCURV_4455"][indxs] * self.columns["EBV"][0]
        self.npts["STIS"] = np.full((len(indxs)), 1)
        self.uncs["STIS"] = spec_dict["EXTCURV_4455_SIG"][indxs] * self.columns["EBV"][0]

        (indxs,) = np.where(1.0 / spec_dict["XVALS"] > 1.0)
        self.waves["BAND"] = (1.0 / spec_dict["XVALS"][indxs]) * u.micron
        self.exts["BAND"] = spec_dict["EXTCURV_4455"][indxs] * self.columns["EBV"][0]
        self.npts["BAND"] = np.full((len(indxs)), 1)
        self.uncs["BAND"] = spec_dict["EXTCURV_4455_SIG"][indxs] * self.columns["EBV"][0]
        self.names["BAND"] = ["JohnJ", "JohnH", "JohnK"]

if __name__ == "__main__":

    fpath = "data/fitzpatrick19/"

    files = glob.glob(f"{fpath}*.save")

    for ifile in files:
        print(ifile)

        ext = ExtData()
        ext.read_ext_data_idlsave(ifile)

        # get A(V) values
        ext.calc_AV_JHK()

        if "AV" in ext.columns.keys():
            ext.calc_RV()

            rv, rv_unc = ext.columns["RV"]
            print("A55", ext.columns["AV"])
            print("R55", rv, rv_unc)
            # exit()

            ext.type = "elx"
            ext.type_rel_band = "V"

            src = "STIS"
            next = rebin_extdata(ext, src, np.array([0.100, 1.05]) * u.micron, 500.)

            ofile = ifile.replace("fitzpatrick19/", "fit19_")
            ofile = ofile.replace("_extcrv.save", "_ext.fits")
            next.save(ofile)
