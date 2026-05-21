import numpy as np
from astropy.table import QTable

from measure_extinction.merge_obsspec import _wavegrid


def rebin_ext(ctable, output_resolution):

    cwaves = ctable["waves"].value
    cfluxes = ctable["d2intercepts"]
    cuncs = ctable["d2intercepts_std"]
    cnpts = np.full(len(cwaves), 1.0)
    crms = ctable["d2rmss"]

    iwave_range = [np.min(cwaves), np.max(cwaves)]
    full_wave, full_wave_min, full_wave_max = _wavegrid(output_resolution, iwave_range)

    n_waves = len(full_wave)
    full_flux = np.zeros((n_waves), dtype=float)
    full_unc = np.zeros((n_waves), dtype=float)
    full_rms = np.zeros((n_waves), dtype=float)
    full_npts = np.zeros((n_waves), dtype=int)

    cnpts[cuncs == 0] = 0
    for k in range(n_waves):
        gvals = (
            (cwaves >= full_wave_min[k]) & (cwaves < full_wave_max[k]) & (cnpts > 0)
        )
        if np.sum(gvals) > 0:
            weights = np.square(1.0 / cuncs[gvals])
            full_flux[k] += np.sum(weights * cfluxes[gvals])
            full_unc[k] += np.sum(weights)
            full_rms[k] += np.sum(crms[gvals])
            full_npts[k] += np.sum(gvals)

    # make sure any wavelengths with no unc are not used
    full_npts[full_unc == 0] = 0

    # divide by the net weights
    (indxs,) = np.where(full_npts > 0)
    if len(indxs) > 0:
        full_flux[indxs] /= full_unc[indxs]
        full_unc[indxs] = np.sqrt(
            1.0 / full_unc[indxs]
        )  # * 1e-10  # put back factor for overflow errors
        full_rms[indxs] /= full_npts[indxs]

    # division by 0.9854 is to fix an issue between the V band photometry and STIS data
    # see G23 Renormalization in the dust_extinction readthedocs
    otable = QTable()
    otable["waves"] = full_wave
    otable["d2intercepts"] = full_flux / 0.9854
    otable["d2intercepts_std"] = full_unc / 0.9854
    otable["d2rmss"] = full_rms / 0.9854

    return otable


if __name__ == "__main__":

    allwaves = []
    allexts = []
    alluncs = []
    allrmss = []
    allorig = []

    files = ["gor09_fuse_irv_params.fits",
             "aiue_iue_irv_params.fits",
             "fit19_stis_irv_params.fits",
             "dec22_spexsxd_irv_params.fits",
             "dec22_spexlxd_irv_params.fits",
             "gor21_irs_irv_params.fits"]
    origins = ["FUSE", "IUE", "STIS/Opt", "SpeX/SXD", "SpeX/LXD", "Spitzer/IRS"]
    for cfile, corig in zip(files, origins):
        itab = QTable.read(f"results/{cfile}")

        # determine average resolution
        delt = np.diff(itab["waves"])
        awave = 0.5 * (itab["waves"][0:-1] + itab["waves"][1:])
        if np.average(awave / delt) > 200.0:
            itab = rebin_ext(itab, 150.0)

        delt = np.diff(itab["waves"])
        awave = 0.5 * (itab["waves"][0:-1] + itab["waves"][1:])
        # print(corig,np.average(awave / delt), len(itab["waves"]))

        gvals  = itab["d2intercepts_std"] > 0.0
        if corig == "FUSE":
            gvals2 = itab["waves"].value > 0.0915
            gvals = gvals * gvals2
        if corig == "IUE":
            gvals2 = np.absolute(itab["waves"].value - 0.1216) > 0.0026
            gvals3 = np.absolute(itab["waves"].value - 0.1539) > 0.0024
            gvals4 = np.absolute(itab["waves"].value - 0.1389) > 0.0019
            gvals = gvals * gvals2 * gvals3 * gvals4

        allwaves.append(itab["waves"][gvals])
        allexts.append(itab["d2intercepts"][gvals])
        alluncs.append(itab["d2intercepts_std"][gvals])
        allrmss.append(itab["d2rmss"][gvals])
        allorig.append([corig] * len(allwaves[-1]))

        # add a floor of 1% uncertainties to deal with *tiny* uncs in the optical
        gvals = (alluncs[-1] / allexts[-1]) < 0.01
        alluncs[-1][gvals] = 0.01 * allexts[-1][gvals]

    otab = QTable()
    otab["wave"] = np.concatenate(allwaves)
    otab["A(l)/A(V)"] = np.concatenate(allexts)
    otab["unc"] = np.concatenate(alluncs)
    otab["rms"] = np.concatenate(allrmss)
    otab["origin"] = np.concatenate(allorig)

    otab.write(
        "MW_diffuse_Gordon23_ext.dat", format="ascii.commented_header", overwrite=True
    )
