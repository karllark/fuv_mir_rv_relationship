import glob
import numpy as np
import astropy.units as u
# import matplotlib.pyplot as plt

from measure_extinction.extdata import ExtData

from rebin_extdata import rebin_extdata


if __name__ == "__main__":

    fpath = "data/gordon09/"

    files = glob.glob(f"{fpath}*bin.fits")

    for fname in files:
        ifile = fname.replace("_bin", "")
        ext = ExtData(ifile)
        # extfuse = ExtData(fname)

        # replace FUSE info with Gordon09 created binned extinction
        # ext.waves["FUSE"] = extfuse.waves["FUSE"]
        # ext.exts["FUSE"] = extfuse.exts["FUSE"]
        # ext.uncs["FUSE"] = extfuse.uncs["FUSE"]
        # ext.npts["FUSE"] = extfuse.npts["FUSE"]

        # convert from A(lambda)/A(V) back to original E(lambda-V)
        # allows for a new fitting of A(V) based on a powerlaw
        av, av_unc = ext.columns["AV"]
        # ebv, ebv_unc = ext.columns["EBV"]
        rv, rv_unc = ext.columns["RV"]
        # print(rv, rv_unc)
        #
        # if "AV" in ext.columns.keys():
        #     ext.calc_RV()
        #
        # rv, rv_unc = ext.columns["RV"]
        # print(av, av_unc)
        # print(ebv, ebv_unc)
        # print(rv, rv_unc)
        # exit()

        print(ext.columns["AV"], rv, rv_unc)
        print(ext.columns["EBV"])

        for src in ext.waves.keys():
            gvals = ext.npts[src] > 0
            ext.exts[src][gvals] = (ext.exts[src][gvals] - 1.0) * av
            ext.uncs[src][gvals] *= av  # at least get the units right
            # remove the av_unc from ext uncs
            #   this reverses what was done in compute_alav_sub.pro
            # ***not working*** results in neg uncs as av_unc/av too high
            # tvals = np.sqrt(
            #     ((ext.uncs[src][gvals] / ext.exts[src][gvals]) ** 2)
            #     - ((av_unc / av) ** 2)
            # )
            # ext.uncs[src][gvals] = tvals * ext.exts[src][gvals]

        ext.type = "elx"
        ext.type_rel_band = "V"

        next = rebin_extdata(ext, "FUSE", np.array([0.09, 0.120]) * u.micron, 500.)
        next = rebin_extdata(next, "IUE", np.array([0.100, 0.35]) * u.micron, 500.)

        ofile = ifile.replace("gordon09/", "gor09_")
        ofile = ofile.replace("_bin", "")
        next.save(ofile)
