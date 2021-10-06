import glob
from tqdm import tqdm

import numpy as np
from astropy.modeling import models, fitting
from astropy.table import QTable

from measure_extinction.extdata import ExtData


def get_alav(exts, src, wave):
    """
    Get the A(lambda)/A(V) values for a particular wavelength for the sample
    """
    n_exts = len(exts)
    oext = np.full((n_exts, 2), np.nan)
    for i, iext in enumerate(exts):
        if src in iext.waves.keys():
            sindxs = np.argsort(np.absolute(iext.waves[src] - wave))
            if (iext.npts[src][sindxs[0]] > 0) and (iext.exts[src][sindxs[0]] > 0):
                oext[i, 0] = iext.exts[src][sindxs[0]]
                oext[i, 1] = iext.uncs[src][sindxs[0]]
            else:
                oext[i, 0] = np.nan
                oext[i, 1] = np.nan

    return oext


def get_rvs(exts):
    """
    Get the R(V) values from the extinction curve info
    """
    rvs = np.zeros((len(exts), 2))
    for i, iext in enumerate(exts):
        irv = iext.columns["RV"]
        rvs[i, 0] = irv[0]
        rvs[i, 1] = irv[1]
    return rvs


def get_irvs(rvs):
    """
    Compute 1/rvs values (including uncs) from rvs vals
    """
    irvs = np.zeros(rvs.shape)
    irvs[:, 0] = 1 / rvs[:, 0]
    irvs[:, 1] = irvs[:, 0] * (rvs[:, 1] / rvs[:, 0])
    return irvs


def fit_allwaves(exts, src, ofilename):
    """
    Fit all the wavelengths for a sample of curves for the specified data
    """
    # rvs
    rvs = get_rvs(exts)
    irvs = get_irvs(rvs)
    xvals = irvs[:, 0]

    # possible wavelengths
    if src not in exts[0].waves.keys():
        poss_waves = exts[10].waves[src]
    else:
        poss_waves = exts[0].waves[src]

    fit = fitting.LinearLSQFitter()
    line_init = models.Linear1D()

    nwaves = len(poss_waves)
    slopes = np.zeros((nwaves))
    intercepts = np.zeros((nwaves))
    npts = np.zeros(nwaves)
    sigmas = np.zeros(nwaves)
    for k in tqdm(range(nwaves), desc=src):
        rwave = poss_waves[k]
        oexts = get_alav(exts, src, rwave)
        if np.sum(np.isfinite(oexts[:, 0])) > 5:
            npts[k] = np.sum(np.isfinite(oexts[:, 0]))
            yvals = oexts[:, 0]
            gvals = np.isfinite(yvals)
            fitted_line = fit(line_init, xvals[gvals], yvals[gvals])
            slopes[k] = fitted_line.slope.value
            intercepts[k] = fitted_line.intercept.value
            sigmas[k] = np.sqrt(
                np.sum(np.square(yvals[gvals] - fitted_line(xvals[gvals])))
                / (npts[k] - 1)
            )

    # save the fits
    otab = QTable()
    otab["waves"] = poss_waves
    otab["slopes"] = slopes
    otab["intercepts"] = intercepts
    otab["sigmas"] = sigmas
    otab["npts"] = npts
    otab.write(f"results/{ofilename}", overwrite=True)

    return (poss_waves, slopes, intercepts, sigmas, npts)


def get_exts(sampstr):
    """
    Read in the extinction curves and transform E(lambda-V) to A(lambda)/A(V)
    """
    files = glob.glob(f"data/{sampstr}*.fits")
    exts = []
    for cfile in files:
        iext = ExtData(cfile)
        iext.trans_elv_alav()
        exts.append(iext)
    return exts


if __name__ == "__main__":

    exts_gor09 = get_exts("gor09")
    fit_allwaves(exts_gor09, "FUSE", "gor09_fuse_irv_params.fits")
    fit_allwaves(exts_gor09, "IUE", "gor09_iue_irv_params.fits")

    exts_fit19 = get_exts("fit19")
    fit_allwaves(exts_fit19, "STIS", "fit19_stis_irv_params.fits")

    exts_fit19 = get_exts("gor21")
    fit_allwaves(exts_fit19, "IUE", "gor21_iue_irv_params.fits")
    fit_allwaves(exts_fit19, "IRS", "gor21_irs_irv_params.fits")

    exts_dec22 = get_exts("decleir22/")
    fit_allwaves(exts_dec22, "IUE", "dec22_iue_irv_params.fits")
    fit_allwaves(exts_dec22, "SpeX_SXD", "dec22_spexsxd_irv_params.fits")
    fit_allwaves(exts_dec22, "SpeX_LXD", "dec22_spexlxd_irv_params.fits")
