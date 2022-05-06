import glob
import argparse
import matplotlib.pyplot as plt

# from matplotlib.patches import Ellipse
from matplotlib import cm, colors
import numpy as np
from math import sqrt, cos, sin
from matplotlib.patches import Polygon
from scipy.linalg import eigh

import astropy.units as u
from astropy.modeling import models, fitting
from astropy.table import Table
from astropy.stats import sigma_clip

from hyperfit.linfit import LinFit as HFLinFit

from measure_extinction.extdata import ExtData

from fit_irv import get_irvs, get_alav
from helpers import mcfit_cov, mcfit_cov_quad

from fit_full2dcor import lnlike_correlated
import scipy.optimize as op


def plot_exts(exts, rvs, avs, ctype, cwave, psym, label, alpha=0.5):
    oexts = get_alav(exts, ctype, cwave)
    xvals = rvs[:, 0]
    xvals_unc = rvs[:, 1]
    yvals = oexts[:, 0]
    yvals_unc = oexts[:, 1]
    avfrac = avs[:, 1] / avs[:, 0]
    ax[i].plot(
        rvs[:, 0],
        oexts[:, 0],
        psym,
        fillstyle="none",
        label=label,
        alpha=alpha,
    )
    # ax[i].errorbar(
    #     rvs[:, 0],
    #     oexts[:, 0],
    #     xerr=xvals_unc,
    #     yerr=yvals_unc,
    #     fmt=psym,
    #     fillstyle="none",
    #     alpha=0.2,
    # )
    return (xvals, xvals_unc, yvals, yvals_unc, avfrac)


# from Dries' dust_fuse_h2 repository
def cov_ellipse(x, y, cov, num_sigma=1, **kwargs):
    """
    Create an ellipse at the coordinates (x,y), that represents the
    covariance. The style of the ellipse can be adjusted using the
    kwargs.

    Returns
    -------
    ellipse: matplotlib.patches.Ellipse
    """

    position = [x, y]

    if cov[0, 1] != 0:
        # length^2 and orientation of ellipse axes is determined by
        # eigenvalues and vectors, respectively. Eigh is more stable for
        # symmetric / hermitian matrices.
        values, vectors = eigh(cov)
        width, height = np.sqrt(np.abs(values)) * num_sigma * 2
    else:
        width = sqrt(cov[0, 0]) * 2
        height = sqrt(cov[1, 1]) * 2
        vectors = np.array([[1, 0], [0, 1]])

    # I ended up using a Polygon just like Karl's plotting code. The
    # ellipse is buggy when the difference in axes is extreme (1e22). I
    # think it is because even a slight rotation will make the ellipse
    # look extremely strechted, as the extremely long axis (~1e22)
    # rotates into the short coordinate (~1).

    # two vectors representing the axes of the ellipse
    vw = vectors[:, 0] * width / 2
    vh = vectors[:, 1] * height / 2

    # generate corners
    num_corners = 64
    angles = np.linspace(0, 2 * np.pi, num_corners, endpoint=False)
    corners = np.row_stack([position + vw * cos(a) + vh * sin(a) for a in angles])

    return Polygon(corners, **kwargs)


def draw_ellipses(ax, xs, ys, covs, num_sigma=1, sigmas=None, **kwargs):
    for k, (x, y, cov) in enumerate(zip(xs, ys, covs)):
        # if sigmas is not None:
        #     color = cm.viridis(sigmas[k] / 3.0)[0]
        # ax.add_patch(cov_ellipse(x, y, cov, num_sigma, color=color, **kwargs))
        ax.add_patch(cov_ellipse(x, y, cov, num_sigma, **kwargs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rv", help="plot versus R(V)", action="store_true")
    parser.add_argument(
        "--incval04", help="include Valencic et al. 2004", action="store_true"
    )
    # parser.add_argument("--elvebv", help="plot versus E(l-V)/E(B-V)", action="store_true")
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    # test angles given correlation coefficients
    # print(45.0 - np.rad2deg(np.arccos(np.array([-1.0, -0.5, 0.0, 0.5, 1.0]))) / 2.0)
    # exit()

    # read in all the extinction curves
    if args.incval04:
        files_val04 = glob.glob("data/val04*.fits")
        exts_val04 = [ExtData(cfile) for cfile in files_val04]
        psym_val04 = "go"

    files_fit19 = glob.glob("data/fit19*.fits")
    exts_fit19 = [ExtData(cfile) for cfile in files_fit19]
    psym_fit19 = "cv"

    files_dec22 = glob.glob("data/dec22*.fits")
    exts_dec22 = [ExtData(cfile) for cfile in files_dec22]
    psym_dec22 = "r>"

    files_gor09 = glob.glob("data/gor09*.fits")
    exts_gor09 = [ExtData(cfile) for cfile in files_gor09]
    psym_gor09 = "bs"

    files_gor21 = glob.glob("data/gor21*.fits")
    exts_gor21 = [ExtData(cfile) for cfile in files_gor21]
    psym_gor21 = "m^"

    # get R(V) values
    n_gor09 = len(files_gor09)
    rvs_gor09 = np.zeros((n_gor09, 2))
    avs_gor09 = np.zeros((n_gor09, 2))
    for i, iext in enumerate(exts_gor09):
        av = iext.columns["AV"]
        avs_gor09[i, 0] = av[0]
        avs_gor09[i, 1] = av[1]

        irv = iext.columns["RV"]
        rvs_gor09[i, 0] = irv[0]
        rvs_gor09[i, 1] = irv[1]

        iext.trans_elv_alav()

    if args.incval04:
        n_val04 = len(files_val04)
        rvs_val04 = np.zeros((n_val04, 2))
        avs_val04 = np.zeros((n_val04, 2))
        for i, iext in enumerate(exts_val04):
            av = iext.columns["AV"]
            avs_val04[i, 0] = av[0]
            avs_val04[i, 1] = av[1]

            irv = iext.columns["RV"]
            rvs_val04[i, 0] = irv[0]
            rvs_val04[i, 1] = irv[1]
            iext.trans_elv_alav()

    # get R(V) values
    n_fit19 = len(files_fit19)
    rvs_fit19 = np.zeros((n_fit19, 2))
    avs_fit19 = np.zeros((n_fit19, 2))
    for i, iext in enumerate(exts_fit19):
        av = iext.columns["AV"]
        avs_fit19[i, 0] = av[0]
        avs_fit19[i, 1] = av[1]

        irv = iext.columns["RV"]
        rvs_fit19[i, 0] = irv[0]
        rvs_fit19[i, 1] = irv[1]
        iext.trans_elv_alav()

    # get R(V) values
    n_gor21 = len(files_gor21)
    rvs_gor21 = np.zeros((n_gor21, 2))
    avs_gor21 = np.zeros((n_gor21, 2))
    for i, iext in enumerate(exts_gor21):
        av = iext.columns["AV"]
        avs_gor21[i, 0] = av[0]
        avs_gor21[i, 1] = 0.5 * (av[1] + av[2])

        irv = iext.columns["RV"]
        rvs_gor21[i, 0] = irv[0]
        rvs_gor21[i, 1] = irv[1]
        iext.trans_elv_alav()

    # get R(V) values
    n_dec22 = len(files_dec22)
    rvs_dec22 = np.zeros((n_dec22, 2))
    avs_dec22 = np.zeros((n_dec22, 2))
    for i, iext in enumerate(exts_dec22):
        av = iext.columns["AV"]
        avs_dec22[i, 0] = av[0]
        avs_dec22[i, 1] = 0.5 * (av[1] + av[2])

        irv = iext.columns["RV"]
        rvs_dec22[i, 0] = irv[0]
        rvs_dec22[i, 1] = 0.5 * (irv[1] + irv[2])
        iext.trans_elv_alav()

    if args.rv:
        labx = "$R(V)$"
        xrange = [2.25, 6.5]
    else:
        if args.incval04:
            rvs_val04 = get_irvs(rvs_val04)
        rvs_gor09 = get_irvs(rvs_gor09)
        rvs_fit19 = get_irvs(rvs_fit19)
        rvs_gor21 = get_irvs(rvs_gor21)
        rvs_dec22 = get_irvs(rvs_dec22)
        labx = "$1/R(V)$ - 1/3.1"
        # xrange = np.array([1.0 / 6.5, 1.0 / 2.0]) - 1 / 3.1
        xrange = np.array([1.0 / 6.5, 1.0 / 2.25]) - 1 / 3.1

    laby = r"$A(\lambda)/A(V)$"

    fontsize = 12

    font = {"size": fontsize}

    plt.rc("font", **font)

    plt.rc("lines", linewidth=1)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("ytick.major", width=2)

    fig, fax = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(12, 12),
        sharex=True,  # constrained_layout=True
    )
    ax = fax.flatten()

    repwaves = {
        "FUSE1": 0.1 * u.micron,
        # "FUSE1": 0.09042863547801971 * u.micron,
        "IUE1": 0.15 * u.micron,
        "IUE2": 0.2175 * u.micron,
        # "IUE3": 0.3 * u.micron,
        "STIS1": 0.45 * u.micron,
        "STIS2": 0.7 * u.micron,
        # "BAND1": 0.45 * u.micron,
        # "BAND2": 2.1 * u.micron,
        "SpeX_SXD1": 1.65 * u.micron,
        "SpeX_LXD1": 3.5 * u.micron,
        "IRS1": 10.0 * u.micron,
        "IRS2": 15.0 * u.micron,
    }

    for i, rname in enumerate(repwaves.keys()):
        xvals = None
        yvals = None
        xvals_unc = None
        yvals_unc = None
        avfrac = None

        print(rname)

        if "FUSE" in rname:
            xvals, xvals_unc, yvals, yvals_unc, avfrac = plot_exts(
                exts_gor09,
                rvs_gor09,
                avs_gor09,
                "FUSE",
                repwaves[rname],
                psym_gor09,
                "G09",
            )

        if "STIS" in rname:
            xvals, xvals_unc, yvals, yvals_unc, avfrac = plot_exts(
                exts_fit19,
                rvs_fit19,
                avs_fit19,
                "STIS",
                repwaves[rname],
                psym_fit19,
                "F19",
            )

        elif "SpeX_SXD" in rname:
            xvals, xvals_unc, yvals, yvals_unc, avfrac = plot_exts(
                exts_dec22,
                rvs_dec22,
                avs_dec22,
                "SpeX_SXD",
                repwaves[rname],
                psym_dec22,
                "D22",
            )

        elif "SpeX_LXD" in rname:
            xvals, xvals_unc, yvals, yvals_unc, avfrac = plot_exts(
                exts_dec22,
                rvs_dec22,
                avs_dec22,
                "SpeX_LXD",
                repwaves[rname],
                psym_dec22,
                "D22",
            )

        elif "IRS" in rname:
            xvals, xvals_unc, yvals, yvals_unc, avfrac = plot_exts(
                exts_gor21,
                rvs_gor21,
                avs_gor21,
                "IRS",
                repwaves[rname],
                psym_gor21,
                "G21",
            )

        elif "IUE" in rname:
            xvals1, xvals1_unc, yvals1, yvals1_unc, avfrac1 = plot_exts(
                exts_gor09,
                rvs_gor09,
                avs_gor09,
                "IUE",
                repwaves[rname],
                psym_gor09,
                "G09",
            )
            xvals2, xvals2_unc, yvals2, yvals2_unc, avfrac2 = plot_exts(
                exts_fit19,
                rvs_fit19,
                avs_fit19,
                "STIS",
                repwaves[rname],
                psym_fit19,
                "F19",
            )
            xvals3, xvals3_unc, yvals3, yvals3_unc, avfrac3 = plot_exts(
                exts_gor21,
                rvs_gor21,
                avs_gor21,
                "IUE",
                repwaves[rname],
                psym_gor21,
                "G21",
            )
            xvals4, xvals4_unc, yvals4, yvals4_unc, avfrac4 = plot_exts(
                exts_dec22,
                rvs_dec22,
                avs_dec22,
                "IUE",
                repwaves[rname],
                psym_dec22,
                "D22",
            )
            xvals = np.concatenate((xvals1, xvals2, xvals3, xvals4))
            xvals_unc = np.concatenate((xvals1_unc, xvals2_unc, xvals3_unc, xvals4_unc))
            yvals = np.concatenate((yvals1, yvals2, yvals3, yvals4))
            yvals_unc = np.concatenate((yvals1_unc, yvals2_unc, yvals3_unc, yvals4_unc))
            avfrac = np.concatenate((avfrac1, avfrac2, avfrac3, avfrac4))

            if args.incval04:
                xvals5, xvals5_unc, yvals5, yvals5_unc, avfrac5 = plot_exts(
                    exts_val04,
                    rvs_val04,
                    avs_val04,
                    "IUE",
                    repwaves[rname],
                    psym_val04,
                    "V04",
                    alpha=0.0,
                )
                xvals = np.concatenate((xvals, xvals5))
                xvals_unc = np.concatenate((xvals_unc, xvals5_unc))
                yvals = np.concatenate((yvals, yvals5))
                yvals_unc = np.concatenate((yvals_unc, yvals5_unc))
                avfrac = np.concatenate((avfrac, avfrac5))

        elif "BAND" in rname:
            # oexts = get_alav(exts_val04, "BAND", repwaves[rname])
            # ax[i].plot(
            #     rvs_val04[:, 0],
            #     oexts[:, 0],
            #     psym_val04,
            #     fillstyle="none",
            #     alpha=0.5,
            #     label="V04",
            # )
            oexts = get_alav(exts_gor09, "BAND", repwaves[rname])
            ax[i].plot(
                rvs_gor09[:, 0], oexts[:, 0], psym_gor09, fillstyle="none", label="G09"
            )
            oexts = get_alav(exts_gor21, "BAND", repwaves[rname])
            ax[i].plot(
                rvs_gor21[:, 0], oexts[:, 0], psym_gor21, fillstyle="none", label="G21"
            )
            oexts = get_alav(exts_dec22, "BAND", repwaves[rname])
            ax[i].plot(
                rvs_dec22[:, 0], oexts[:, 0], psym_dec22, fillstyle="none", label="D22"
            )
        ax[i].legend(title=f"{repwaves[rname]}", ncol=2)

        # save the data
        a = Table()
        gvals = np.isfinite(yvals)
        a["irv"] = xvals[gvals]
        a["irv_unc"] = xvals_unc[gvals]
        a["alav"] = yvals[gvals]
        a["alav_unc"] = yvals_unc[gvals]
        a.write(
            f"results/fuv_mir_data_{rname}_{repwaves[rname].value}.dat",
            format="ascii.commented_header",
            overwrite=True,
        )

        # xvals = None

        do_hfit = False
        do_mcfit = False

        # fit a line
        if xvals is not None:
            fit = fitting.LinearLSQFitter()
            line_init = models.Linear1D()
            gvals = np.isfinite(yvals)
            # print(np.sum(gvals))
            # gvals[xvals_unc > 0.05] = False
            # print(np.sum(gvals))
            # print("max 1/rv unc used", max(xvals_unc[gvals]))
            fitted_line = fit(
                line_init,
                xvals[gvals],
                yvals[gvals],  # , weights=1.0 / yvals_unc[gvals]
            )
            mod_xvals = np.arange(xrange[0], xrange[1], 0.01)

            or_fit = fitting.FittingWithOutlierRemoval(
                fit, sigma_clip, niter=3, sigma=3.0
            )
            fitted_line, mask = or_fit(
                line_init,
                xvals[gvals],
                yvals[gvals],  # , weights=1.0 / yvals_unc[gvals]
            )
            not_mask = np.logical_not(mask)
            bad_data = np.ma.masked_array(yvals[gvals], mask=not_mask)
            ax[i].plot(xvals[gvals], bad_data, "rx")

            ax[i].plot(
                mod_xvals, fitted_line(mod_xvals), "k--", label="Fit", alpha=0.75, lw=3
            )

            # quadratic fit
            quad_init = models.Polynomial1D(2)
            fitted_quad, mask = or_fit(
                quad_init,
                xvals[gvals],
                yvals[gvals],  # , weights=1.0 / yvals_unc[gvals]
            )
            ax[i].plot(
                mod_xvals,
                fitted_quad(mod_xvals),
                "k:",
                label="Quad Fit",
                alpha=0.75,
                lw=3,
            )

            ndata = np.sum(gvals)
            # avfrac = np.full(len(xvals), 0.0)
            # xvals_unc = xvals_unc / 2.0
            # yvals_unc = yvals_unc / 2.0

            # linear approximation - can result in > 1 correlation coefficients
            cov_xy = (xvals[gvals] + 1 / 3.1) * yvals[gvals] * (avfrac[gvals] ** 2)
            corr_xy = cov_xy / (xvals_unc[gvals] * yvals_unc[gvals])
            corr_xy[corr_xy > 0.9] = 0.9

            covs = np.zeros((ndata, 2, 2))
            for k in range(ndata):
                kk = k
                covs[k, 0, 0] = xvals_unc[gvals][kk] ** 2
                covs[k, 0, 1] = (
                    corr_xy[kk] * xvals_unc[gvals][kk] * yvals_unc[gvals][kk]
                )
                covs[k, 1, 0] = (
                    corr_xy[kk] * xvals_unc[gvals][kk] * yvals_unc[gvals][kk]
                )
                covs[k, 1, 1] = yvals_unc[gvals][kk] ** 2

            draw_ellipses(
                ax[i], xvals[gvals], yvals[gvals], covs, color="black", alpha=0.1
            )

            # fit with new full 2D fitting
            def nll(*args):
                return -lnlike_correlated(*args)

            params = fitted_quad.parameters
            print("start", params)
            intinfo = [-0.20, 0.20, 0.0001]
            result = op.minimize(
                nll,
                params,
                args=(yvals[gvals], fitted_quad, covs, intinfo, xvals[gvals]),
            )
            nparams = result["x"]
            print(nparams)

            fitted_quad.parameters = nparams
            ax[i].plot(mod_xvals, fitted_quad(mod_xvals), "p-", alpha=0.5, lw=3)

            if do_mcfit:

                nummc = 20
                mcparams = mcfit_cov(
                    xvals[gvals], yvals[gvals], covs, not_mask, num=nummc, ax=ax[i]
                )

                for k in range(nummc):
                    fitted_line.intercept = mcparams[k, 0]
                    fitted_line.slope = mcparams[k, 1]
                    ax[i].plot(mod_xvals, fitted_line(mod_xvals), "b-", alpha=0.5, lw=3)

                mcparams = mcfit_cov_quad(
                    xvals[gvals],
                    yvals[gvals],
                    covs,
                    not_mask,
                    num=nummc,
                    ax=ax[i],
                    # quad_init=quad_init,
                )

                fitted_quad = models.Polynomial1D(2)
                for k in range(nummc):
                    fitted_quad.c0 = mcparams[k, 0]
                    fitted_quad.c1 = mcparams[k, 1]
                    fitted_quad.c2 = mcparams[k, 2]
                    ax[i].plot(mod_xvals, fitted_quad(mod_xvals), "g-", alpha=0.5, lw=3)

            if do_hfit:
                # only fit the non-rejected points
                xvals_good = xvals[gvals][not_mask]
                xvals_unc_good = xvals_unc[gvals][not_mask]
                yvals_good = yvals[gvals][not_mask]
                yvals_unc_good = yvals_unc[gvals][not_mask]
                covs_good = covs[not_mask, :, :]
                ndata = len(xvals_good)

                # fit a line with hyperfit to account for correlated uncertainties
                hfdata, hfcov = np.zeros((2, ndata)), np.zeros((2, 2, ndata))

                hfdata[0, :] = xvals_good
                hfdata[1, :] = yvals_good
                for k in range(ndata):
                    hfcov[:, :, k] = covs_good[k, :, :]

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
                # print(bounds)
                # bounds = ((-2.0, 50.0), (-5.0, 30.0), (1.0e-5, 5.0))
                bounds = ((-5.0, 30.0), (-1.0, 20.0), (1.0e-5, 5.0))
                # hf_fit_params = hf_fit.optimize(bounds, verbose=False)
                mcmc_samples, mcmc_lnlike = hf_fit.emcee(bounds, verbose=False)
                print(np.mean(mcmc_samples, axis=1), np.std(mcmc_samples, axis=1))
                mean_params = np.mean(mcmc_samples, axis=1)
                mean_stds = np.std(mcmc_samples, axis=1)
                # mod_yvals_hf = hf_fit.coords[1] + hf_fit.coords[0] * mod_xvals
                mod_yvals_hf = mean_params[1] + mean_params[0] * mod_xvals
                ax[i].plot(mod_xvals, mod_yvals_hf, "k-", label="HF Fit")

                ax[i].plot(mod_xvals, mod_yvals_hf - hf_fit.vert_scat, "k--")
                ax[i].plot(mod_xvals, mod_yvals_hf + hf_fit.vert_scat, "k--")

                # print(hf_fit_params)

                sigmas = hf_fit.get_sigmas()

                # draw_ellipses(
                #     ax[i], xvals_good, yvals_good, covs_good, sigmas=sigmas, alpha=0.15
                # )

    fax[0, 1].set_xlim(xrange)

    # for 2nd x-axis with R(V) values
    axis_rvs = np.array([2.3, 2.5, 3.0, 4.0, 5.0, 6.0])
    new_ticks = 1 / axis_rvs - 1 / 3.1
    new_ticks_labels = ["%.1f" % z for z in axis_rvs]

    for i in range(3):
        fax[2, i].set_xlabel(labx)

        if not args.rv:
            # add 2nd x-axis with R(V) values
            tax = fax[0, i].twiny()
            tax.set_xlim(fax[0, i].get_xlim())
            tax.set_xticks(new_ticks)
            tax.set_xticklabels(new_ticks_labels)
            tax.set_xlabel(r"$R(V)$")

    for i in range(3):
        fax[i, 0].set_ylabel(laby)

    # Add the colourbar
    cb = fig.colorbar(
        cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=3.0), cmap=cm.viridis),
        ax=fax,
        shrink=0.12,
        alpha=0.5,
        aspect=10,
        anchor=(-4.7, 0.85),
    )
    cb.set_label(label=r"$\sigma$ offset", fontsize=14)

    fig.tight_layout()

    fname = "fuv_mir_rv_rep_waves"
    if args.rv:
        fname = f"{fname}_rv"
    if args.png:
        fig.savefig(f"{fname}.png")
    elif args.pdf:
        fig.savefig(f"{fname}.pdf")
    else:
        plt.show()
