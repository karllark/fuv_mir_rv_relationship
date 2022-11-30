import glob
import argparse
from tqdm import tqdm

import numpy as np
from astropy.modeling import models, fitting
from astropy.table import QTable

# from astropy.stats import sigma_clip
from hyperfit.linfit import LinFit as HFLinFit

from measure_extinction.extdata import ExtData

from helpers import mcfit_cov
from fit_full2dcor import fit_2Dcorrelated, fit_2Dcorrelated_emcee


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


def get_avs(exts):
    """
    Get the A(V) values from the extinction curve info
    """
    avs = np.zeros((len(exts), 2))
    for i, iext in enumerate(exts):
        av = iext.columns["AV"]
        avs[i, 0] = av[0]
        avs[i, 1] = av[1]
    return avs


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
    1/rvs have 1/3.1 subtracted for the fitting
    """
    irvs = np.zeros(rvs.shape)
    irvs[:, 0] = 1 / rvs[:, 0]
    irvs[:, 1] = irvs[:, 0] * (rvs[:, 1] / rvs[:, 0])
    irvs[:, 0] -= 1 / 3.1
    return irvs


def get_best_fit_params(sampler):
    """
    Determine the best fit parameters given an emcee sampler object
    """
    # very likely a faster way
    max_lnp = -1e6
    nwalkers, nsteps = sampler.lnprobability.shape
    for k in range(nwalkers):
        tmax_lnp = np.nanmax(sampler.lnprobability[k])
        if tmax_lnp > max_lnp:
            max_lnp = tmax_lnp
            (indxs,) = np.where(sampler.lnprobability[k] == tmax_lnp)
            fit_params_best = sampler.chain[k, indxs[0], :]

    return fit_params_best


def fit_allwaves(
    exts,
    src,
    ofilename,
    hfemcee=False,
    do_hfit=False,
    do_mcfit=False,
    do_2dfit=True,
    do_2dfit_emcee=False,
    save_chains=False,
):
    """
    Fit all the wavelengths for a sample of curves for the specified data
    """
    # rvs
    rvs = get_rvs(exts)
    irvs = get_irvs(rvs)
    xvals = irvs[:, 0]
    xvals_unc = irvs[:, 1]

    # avs
    avs = get_avs(exts)
    avfrac = avs[:, 1] / avs[:, 0]

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
    rmss = np.zeros(nwaves)

    d2slopes = np.zeros((nwaves))
    d2slopes_unc = np.zeros((nwaves))
    d2intercepts = np.zeros((nwaves))
    d2intercepts_unc = np.zeros((nwaves))
    d2rmss = np.zeros((nwaves))
    d2lnlikes = np.zeros((nwaves))
    d2curves_quad = np.zeros((nwaves))
    d2slopes_quad = np.zeros((nwaves))
    d2intercepts_quad = np.zeros((nwaves))
    d2rmss_quad = np.zeros((nwaves))
    d2lnlikes_quad = np.zeros((nwaves))

    mcslopes = np.zeros((nwaves))
    mcslopes_unc = np.zeros((nwaves))
    mcintercepts = np.zeros((nwaves))
    mcintercepts_unc = np.zeros((nwaves))

    hfslopes = np.zeros((nwaves))
    hfintercepts = np.zeros((nwaves))
    hfsigmas = np.zeros(nwaves)
    hfslopes_std = np.zeros((nwaves))
    hfintercepts_std = np.zeros((nwaves))
    hfsigmas_std = np.zeros(nwaves)
    hfrmss = np.zeros(nwaves)
    for k in tqdm(range(nwaves), desc=src):
        rwave = poss_waves[k]
        oexts = get_alav(exts, src, rwave)
        if np.sum(np.isfinite(oexts[:, 0])) > 5:
            # print(rwave)
            # regular weighted fit
            npts[k] = np.sum(np.isfinite(oexts[:, 0]))
            yvals = oexts[:, 0]
            yvals_unc = oexts[:, 1]
            gvals = np.isfinite(yvals)
            fitted_line = fit(
                line_init, xvals[gvals], yvals[gvals], weights=1.0 / yvals_unc[gvals]
            )

            # or_fit = fitting.FittingWithOutlierRemoval(
            #     fit, sigma_clip, niter=3, sigma=3.0
            # )
            # fitted_line, mask = or_fit(
            #     line_init,
            #     xvals[gvals],
            #     yvals[gvals],  # , weights=1.0 / yvals_unc[gvals]
            # )
            # not_mask = np.logical_not(mask)
            not_mask = np.full(len(xvals[gvals]), True)

            # print(fitted_line)
            slopes[k] = fitted_line.slope.value
            intercepts[k] = fitted_line.intercept.value
            rmss[k] = np.sqrt(
                np.sum(np.square(yvals[gvals] - fitted_line(xvals[gvals])))
                / (npts[k] - 1)
            )

            # define needed covariance information
            ndata = np.sum(gvals)
            # linear approximation - can result in > 1 correlation coefficients
            # xvals_unc[gvals] /= 4.0
            cov_xy = (xvals[gvals] + 1 / 3.1) * yvals[gvals] * (avfrac[gvals] ** 2)
            corr_xy = cov_xy / (xvals_unc[gvals] * yvals_unc[gvals])
            # put a max on the correlation coefficient
            max_corr = 0.99
            corr_xy[corr_xy > max_corr] = max_corr

            covs = np.zeros((ndata, 2, 2))
            for kk in range(ndata):
                covs[kk, 0, 0] = xvals_unc[gvals][kk] ** 2
                covs[kk, 0, 1] = (
                    corr_xy[kk] * xvals_unc[gvals][kk] * yvals_unc[gvals][kk]
                )
                covs[kk, 1, 0] = (
                    corr_xy[kk] * xvals_unc[gvals][kk] * yvals_unc[gvals][kk]
                )
                covs[kk, 1, 1] = yvals_unc[gvals][kk] ** 2

            # do the 2D covariance fitting
            if do_2dfit:
                # fit with new full 2D fitting (use unweigthed linear fit to start)
                intinfo = [-0.20, 0.20, 0.0001]
                if save_chains:
                    chain_filename = f"results/chains/{src}_{rwave.value}.h5"
                else:
                    chain_filename = None
                if do_2dfit_emcee:
                    nsteps = 1000
                    fit2d_line = fit_2Dcorrelated_emcee(
                        xvals[gvals],
                        yvals[gvals],
                        covs,
                        fitted_line,
                        intinfo,
                        nsteps=nsteps,
                        sfilename=chain_filename,
                    )
                    # bparams = get_best_fit_params(fit2d_line.sampler)

                    samples = fit2d_line.sampler.get_chain(
                        flat=True, discard=int(0.1 * nsteps)
                    )

                    d2slopes[k] = np.mean(samples[:, 0])
                    d2slopes_unc[k] = np.std(samples[:, 0])
                    d2intercepts[k] = np.mean(samples[:, 1])
                    d2intercepts_unc[k] = np.std(samples[:, 1])

                    # print(intercepts[k], d2intercepts[k], d2intercepts_unc[k])
                    # print(slopes[k], d2slopes[k], d2slopes_unc[k])
                    # exit()

                    # d2lnlikes[k] = -1.0 * fit2d_line.result["fun"]
                else:
                    fit2d_line = fit_2Dcorrelated(
                        xvals[gvals], yvals[gvals], covs, fitted_line, intinfo
                    )
                    d2slopes[k] = fit2d_line.slope.value
                    d2intercepts[k] = fit2d_line.intercept.value
                    d2rmss[k] = np.sqrt(
                        np.sum(np.square(yvals[gvals] - fit2d_line(xvals[gvals])))
                        / (npts[k] - 1)
                    )
                    d2lnlikes[k] = -1.0 * fit2d_line.result["fun"]

                # initial unweighted quadratic fit
                quad_init = models.Polynomial1D(2)
                fitted_quad = fit(quad_init, xvals[gvals], yvals[gvals])

                # full 2D corrlated quad fit
                fit2d_quad = fit_2Dcorrelated(
                    xvals[gvals], yvals[gvals], covs, fitted_quad, intinfo
                )
                d2curves_quad[k] = fit2d_quad.c2.value
                d2slopes_quad[k] = fit2d_quad.c1.value
                d2intercepts_quad[k] = fit2d_quad.c0.value
                d2rmss_quad[k] = np.sqrt(
                    np.sum(np.square(yvals[gvals] - fit2d_quad(xvals[gvals])))
                    / (npts[k] - 1)
                )
                d2lnlikes_quad[k] = -1.0 * fit2d_quad.result["fun"]

            # do Monte Carlo fitting if asked
            if do_mcfit:
                nummc = 1000
                mcparams = mcfit_cov(
                    xvals[gvals], yvals[gvals], covs, not_mask, num=nummc
                )
                mcslopes[k] = np.mean(mcparams[:, 1])
                mcslopes_unc[k] = np.std(mcparams[:, 1])
                mcintercepts[k] = np.mean(mcparams[:, 0])
                mcintercepts_unc[k] = np.std(mcparams[:, 0])
                # print(k, mcslopes[k], mcintercepts[k])

            if do_hfit:
                # only fit the non-rejected points
                xvals_good = xvals[gvals][not_mask]
                yvals_good = yvals[gvals][not_mask]
                covs_good = covs[not_mask, :, :]
                ndata = len(xvals_good)

                # fit a line with hyperfit to account for correlated uncertainties
                hfdata, hfcov = np.zeros((2, ndata)), np.zeros((2, 2, ndata))
                hfdata[0, :] = xvals_good
                hfdata[1, :] = yvals_good
                for k in range(ndata):
                    hfcov[:, :, k] = covs_good[k, :, :]

                # print(hfdata)
                # print(hfcov)
                # exit()

                hf_fit = HFLinFit(hfdata, hfcov)

                # ds = 0.5 * np.absolute(fitted_line.slope)
                # di = 0.5 * np.absolute(fitted_line.intercept)
                # ds = 5.0
                # di = 5.0
                # bounds = (
                #     (fitted_line.slope - ds, fitted_line.slope + ds),
                #     (fitted_line.intercept - di, fitted_line.intercept + di),
                #     (1.0e-5, 5.0),
                # )
                bounds = ((-30.0, 30.0), (-1.0, 20.0), (1.0e-5, 5.0))
                if not hfemcee:
                    hf_fit_params = hf_fit.optimize(bounds, verbose=False)
                    # print(hf_fit_params)
                    hfslopes[k] = hf_fit_params[0][0]
                    hfintercepts[k] = hf_fit_params[0][1]
                    hfsigmas[k] = hf_fit_params[1]
                    # print(fitted_line)
                    # print(bounds)
                else:
                    mcmc_samples, mcmc_lnlike = hf_fit.emcee(bounds, verbose=False)
                    # print(np.mean(mcmc_samples, axis=1), np.std(mcmc_samples, axis=1))
                    mean_params = np.mean(mcmc_samples, axis=1)
                    mean_stds = np.std(mcmc_samples, axis=1)
                    hfslopes[k] = mean_params[0]
                    hfintercepts[k] = mean_params[1]
                    hfsigmas[k] = mean_params[2]
                    hfslopes_std[k] = mean_stds[0]
                    hfintercepts_std[k] = mean_stds[1]
                    hfsigmas_std[k] = mean_stds[2]

                hf_modline = hfintercepts[k] + hfslopes[k] * xvals[gvals]
                hfrmss[k] = np.sqrt(
                    np.sum(np.square(yvals[gvals] - hf_modline)) / (npts[k] - 1)
                )

    # save the fits
    otab = QTable()
    otab["waves"] = poss_waves
    otab["slopes"] = slopes
    otab["intercepts"] = intercepts
    otab["rmss"] = rmss
    otab["npts"] = npts

    if do_2dfit:
        otab["d2slopes"] = d2slopes
        otab["d2intercepts"] = d2intercepts
        otab["d2slopes_std"] = d2slopes_unc
        otab["d2intercepts_std"] = d2intercepts_unc
        otab["d2rmss"] = d2rmss
        otab["d2lnlikes"] = d2lnlikes
        otab["d2curves_quad"] = d2curves_quad
        otab["d2slopes_quad"] = d2slopes_quad
        otab["d2rmss_quad"] = d2rmss_quad
        otab["d2intercepts_quad"] = d2intercepts_quad
        otab["d2lnlikes_quad"] = d2lnlikes_quad

    if do_mcfit:
        otab["mcslopes"] = mcslopes
        otab["mcintercepts"] = mcintercepts
        otab["mcslopes_std"] = mcslopes_unc
        otab["mcintercepts_std"] = mcintercepts_unc

    if do_hfit:
        otab["hfslopes"] = hfslopes
        otab["hfintercepts"] = hfintercepts
        otab["hfsigmas"] = hfsigmas
        otab["hfrmss"] = hfrmss
        if hfemcee:
            otab["hfslopes_std"] = hfslopes_std
            otab["hfintercepts_std"] = hfintercepts_std
            otab["hfsigmas_std"] = hfsigmas_std

    otab.write(f"results/{ofilename}", overwrite=True)

    return (poss_waves, slopes, intercepts, rmss, npts)


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

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["G09", "F19", "G21", "D22", "AIUE"],
        default="G09",
        help="Dataset to fit",
    )
    args = parser.parse_args()

    hfemcee = False
    do_2dfit_emcee = True

    if args.dataset == "G09":
        exts_gor09 = get_exts("gor09")
        fit_allwaves(
            exts_gor09,
            "FUSE",
            "gor09_fuse_irv_params.fits",
            hfemcee=hfemcee,
            do_2dfit_emcee=do_2dfit_emcee,
        )
        # fit_allwaves(exts_gor09, "IUE", "gor09_iue_irv_params.fits", hfemcee=hfemcee)
    elif args.dataset == "F19":
        exts_fit19 = get_exts("fit19")
        fit_allwaves(
            exts_fit19,
            "STIS",
            "fit19_stis_irv_params.fits",
            hfemcee=hfemcee,
            do_2dfit_emcee=do_2dfit_emcee,
        )
    elif args.dataset == "G21":
        exts_gor21 = get_exts("gor21")
        # fit_allwaves(exts_gor21, "IUE", "gor21_iue_irv_params.fits", hfemcee=hfemcee)
        fit_allwaves(
            exts_gor21,
            "IRS",
            "gor21_irs_irv_params.fits",
            hfemcee=hfemcee,
            do_2dfit_emcee=do_2dfit_emcee,
        )
    elif args.dataset == "D22":
        exts_dec22 = get_exts("dec22")
        # fit_allwaves(exts_dec22, "IUE", "dec22_iue_irv_params.fits", hfemcee=hfemcee)
        fit_allwaves(
            exts_dec22,
            "SpeX_SXD",
            "dec22_spexsxd_irv_params.fits",
            hfemcee=hfemcee,
            do_2dfit_emcee=do_2dfit_emcee,
        )
        fit_allwaves(
            exts_dec22,
            "SpeX_LXD",
            "dec22_spexlxd_irv_params.fits",
            hfemcee=hfemcee,
            do_2dfit_emcee=do_2dfit_emcee,
        )
    elif args.dataset == "AIUE":
        exts_gor09 = get_exts("gor09")
        exts_fit19 = get_exts("fit19")
        exts_gor21 = get_exts("gor21")
        exts_dec22 = get_exts("dec22")
        all_exts = exts_gor09 + exts_fit19 + exts_gor21 + exts_dec22
        fit_allwaves(all_exts, "IUE", "aiue_iue_irv_params.fits", hfemcee=hfemcee)
