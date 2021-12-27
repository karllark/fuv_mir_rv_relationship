import glob
import numpy as np
from scipy.io import readsav

# import matplotlib.pyplot as plt
import astropy.units as u

from measure_extinction.extdata import ExtData as ExtDataStock

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
            spec_dict["EXTCURV_RAW_SIG"][sindxs[0]],
        )

        (indxs,) = np.where(1.0 / spec_dict["XVALS"] < 1.0)
        self.waves["STIS"] = (1.0 / spec_dict["XVALS"][indxs]) * u.micron
        self.exts["STIS"] = spec_dict["EXTCURV_RAW"][indxs]
        self.npts["STIS"] = spec_dict["EXTCURV_RAW_SIG"][indxs]
        self.uncs["STIS"] = spec_dict["EXTCURV_RAW_SIG"][indxs]

        # print(self.columns["EBV"])
        # plt.plot(self.waves["STIS"], self.exts["STIS"], label="raw")
        # plt.plot(self.waves["STIS"], spec_dict["EXTCURV_4455"][indxs])
        # plt.legend()
        # plt.show()

        (indxs,) = np.where(1.0 / spec_dict["XVALS"] > 1.0)
        self.waves["BAND"] = (1.0 / spec_dict["XVALS"][indxs]) * u.micron
        self.exts["BAND"] = spec_dict["EXTCURV_RAW"][indxs]
        self.npts["BAND"] = spec_dict["EXTCURV_RAW_SIG"][indxs]
        self.uncs["BAND"] = spec_dict["EXTCURV_RAW_SIG"][indxs]
        self.names["BAND"] = ["JohnJ", "JohnH", "JohnK"]


if __name__ == "__main__":

    fpath = "data/fitzpatrick19/"

    files = glob.glob(f"{fpath}*.save")

    for ifile in files:

        ext = ExtData()
        ext.read_ext_data_idlsave(ifile)

        # get A(V) values
        ext.calc_AV_JHK()
        print(ifile, ext.columns["AV"])

        if "AV" in ext.columns.keys():
            ext.calc_RV()

            ext.type = "elx"
            ext.type_rel_band = "V"

            ofile = ifile.replace("fitzpatrick19/", "fit19_")
            ofile = ofile.replace("_extcrv.save", "_ext.fits")
            ext.save(ofile)
