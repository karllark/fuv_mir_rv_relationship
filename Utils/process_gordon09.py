import glob
# import numpy as np

from measure_extinction.extdata import ExtData

if __name__ == "__main__":

    fpath = "data/gordon09/"

    # files = ["bd+35d4258_hd097471_ext.fits"]
    files = glob.glob(f"{fpath}*bin.fits")

    for fname in files:
        ifile = fname.replace("_bin", "")
        ext = ExtData(ifile)

        # convert from A(lambda)/A(V) back to original E(lambda-V)
        # allows for a new fitting of A(V) based on a powerlaw
        av, av_unc = ext.columns["AV"]
        # ebv, ebv_unc = ext.columns["EBV"]
        # rv, rv_unc = ext.columns["RV"]
        for src in ext.waves.keys():
            gvals = ext.npts[src] > 0
            ext.exts[src][gvals] = (ext.exts[src][gvals] - 1.0) * av
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

        ofile = ifile.replace("gordon09/", "gor09_")
        ext.save(ofile)