import glob
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import cm
import numpy as np

import astropy.units as u
from astropy.modeling import models, fitting

from hyperfit.linfit import LinFit as HFLinFit

from measure_extinction.extdata import ExtData

from fit_irv import get_irvs, get_alav


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rv", help="plot versus R(V)", action="store_true")
    # parser.add_argument("--elvebv", help="plot versus E(l-V)/E(B-V)", action="store_true")
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    # read in all the extinction curves
    files_val04 = glob.glob("data/val04*.fits")
    exts_val04 = [ExtData(cfile) for cfile in files_val04]
    psym_val04 = "go"

    files_gor09 = glob.glob("data/gor09*.fits")
    exts_gor09 = [ExtData(cfile) for cfile in files_gor09]
    psym_gor09 = "bs"

    files_fit19 = glob.glob("data/fit19*.fits")
    exts_fit19 = [ExtData(cfile) for cfile in files_fit19]
    psym_fit19 = "cv"

    files_gor21 = glob.glob("data/gor21*.fits")
    exts_gor21 = [ExtData(cfile) for cfile in files_gor21]
    psym_gor21 = "m^"

    files_dec22 = glob.glob("data/decleir22/*.fits")
    exts_dec22 = [ExtData(cfile) for cfile in files_dec22]
    psym_dec22 = "r>"

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

    n_val04 = len(files_val04)
    rvs_val04 = np.zeros((n_val04, 2))
    for i, iext in enumerate(exts_val04):
        irv = iext.columns["RV"]
        rvs_val04[i, 0] = irv[0]
        rvs_val04[i, 1] = irv[1]
        iext.trans_elv_alav()

    # get R(V) values
    n_fit19 = len(files_fit19)
    rvs_fit19 = np.zeros((n_fit19, 2))
    for i, iext in enumerate(exts_fit19):
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
        xrange = [2.0, 6.5]
    else:
        rvs_val04 = get_irvs(rvs_val04)
        rvs_gor09 = get_irvs(rvs_gor09)
        rvs_fit19 = get_irvs(rvs_fit19)
        rvs_gor21 = get_irvs(rvs_gor21)
        rvs_dec22 = get_irvs(rvs_dec22)
        labx = "$1/R(V)$"
        xrange = [1.0 / 6.5, 1.0 / 2.0]

    laby = r"$A(\lambda)/A(V)$"

    fontsize = 12

    font = {"size": fontsize}

    plt.rc("font", **font)

    plt.rc("lines", linewidth=1)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("ytick.major", width=2)

    fig, fax = plt.subplots(nrows=2, ncols=4, figsize=(17, 8), sharex=True)
    ax = fax.flatten()

    repwaves = {
        "FUSE1": 0.1 * u.micron,
        "IUE1": 0.15 * u.micron,
        "IUE2": 0.2175 * u.micron,
        # "IUE3": 0.3 * u.micron,
        "STIS1": 0.45 * u.micron,
        "STIS2": 0.7 * u.micron,
        # "BAND1": 0.45 * u.micron,
        # "BAND2": 2.1 * u.micron,
        "SpeX_SXD1": 1.65 * u.micron,
        "SpeX_LXD1": 3.5 * u.micron,
        "IRS1": 15.0 * u.micron,
    }

    for i, rname in enumerate(repwaves.keys()):
        xvals = None
        yvals = None
        xvals_unc = None
        yvals_unc = None
        avfrac = None
        if "FUSE" in rname:
            oexts = get_alav(exts_gor09, "FUSE", repwaves[rname])
            xvals = rvs_gor09[:, 0]
            xvals_unc = rvs_gor09[:, 1]
            yvals = oexts[:, 0]
            yvals_unc = oexts[:, 1]
            avfrac = avs_gor09[:, 1] / avs_gor09[:, 0]
            ax[i].errorbar(
                rvs_gor09[:, 0],
                oexts[:, 0],
                xerr=xvals_unc,
                yerr=yvals_unc,
                fmt=psym_gor09,
                fillstyle="none",
                label="G09",
            )
        if "STIS" in rname:
            oexts = get_alav(exts_fit19, "STIS", repwaves[rname])
            xvals = rvs_fit19[:, 0]
            xvals_unc = rvs_fit19[:, 1]
            yvals = oexts[:, 0]
            yvals_unc = oexts[:, 1]
            ax[i].errorbar(
                rvs_fit19[:, 0],
                oexts[:, 0],
                xerr=xvals_unc,
                yerr=yvals_unc,
                fmt=psym_fit19,
                fillstyle="none",
                label="F19",
            )

        elif "SpeX_SXD" in rname:
            oexts = get_alav(exts_dec22, "SpeX_SXD", repwaves[rname])
            xvals = rvs_dec22[:, 0]
            xvals_unc = rvs_dec22[:, 1]
            yvals = oexts[:, 0]
            yvals_unc = oexts[:, 1]
            avfrac = avs_dec22[:, 1] / avs_dec22[:, 0]
            print(yvals_unc)
            ax[i].errorbar(
                rvs_dec22[:, 0],
                oexts[:, 0],
                xerr=xvals_unc,
                yerr=yvals_unc,
                fmt=psym_dec22,
                fillstyle="none",
                label="D22",
            )
        elif "SpeX_LXD" in rname:
            oexts = get_alav(exts_dec22, "SpeX_LXD", repwaves[rname])
            xvals = rvs_dec22[:, 0]
            xvals_unc = rvs_dec22[:, 1]
            yvals = oexts[:, 0]
            yvals_unc = oexts[:, 1]
            avfrac = avs_dec22[:, 1] / avs_dec22[:, 0]
            ax[i].errorbar(
                rvs_dec22[:, 0],
                oexts[:, 0],
                xerr=xvals_unc,
                yerr=yvals_unc,
                fmt=psym_dec22,
                fillstyle="none",
                label="D22",
            )
        elif "IRS" in rname:
            oexts = get_alav(exts_gor21, "IRS", repwaves[rname])
            xvals = rvs_gor21[:, 0]
            xvals_unc = rvs_gor21[:, 1]
            yvals = oexts[:, 0]
            yvals_unc = oexts[:, 1]
            avfrac = avs_gor21[:, 1] / avs_gor21[:, 0]
            ax[i].errorbar(
                rvs_gor21[:, 0],
                oexts[:, 0],
                xerr=xvals_unc,
                yerr=yvals_unc,
                fmt=psym_gor21,
                fillstyle="none",
                label="G21",
            )
        elif "IUE" in rname:
            oexts = get_alav(exts_gor09, "IUE", repwaves[rname])
            xvals = rvs_gor09[:, 0]
            yvals = oexts[:, 0]
            ax[i].plot(
                rvs_gor09[:, 0], oexts[:, 0], psym_gor09, fillstyle="none", label="G09"
            )
            oexts = get_alav(exts_fit19, "STIS", repwaves[rname])
            xvals = np.append(xvals, rvs_fit19[:, 0])
            yvals = np.append(yvals, oexts[:, 0])
            ax[i].plot(
                rvs_fit19[:, 0], oexts[:, 0], psym_fit19, fillstyle="none", label="F19"
            )
            oexts = get_alav(exts_gor21, "IUE", repwaves[rname])
            xvals = np.append(xvals, rvs_gor21[:, 0])
            yvals = np.append(yvals, oexts[:, 0])
            ax[i].plot(
                rvs_gor21[:, 0], oexts[:, 0], psym_gor21, fillstyle="none", label="G21"
            )
            oexts = get_alav(exts_dec22, "IUE", repwaves[rname])
            xvals = np.append(xvals, rvs_dec22[:, 0])
            yvals = np.append(yvals, oexts[:, 0])
            ax[i].plot(
                rvs_dec22[:, 0], oexts[:, 0], psym_dec22, fillstyle="none", label="D22"
            )
        elif "BAND" in rname:
            oexts = get_alav(exts_val04, "BAND", repwaves[rname])
            ax[i].plot(
                rvs_val04[:, 0],
                oexts[:, 0],
                psym_val04,
                fillstyle="none",
                alpha=0.5,
                label="V04",
            )
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

        # fit a line
        if xvals is not None:
            fit = fitting.LinearLSQFitter()
            line_init = models.Linear1D()
            gvals = np.isfinite(yvals)
            fitted_line = fit(line_init, xvals[gvals], yvals[gvals])

            mod_xvals = np.array(xrange)
            ax[i].plot(mod_xvals, fitted_line(mod_xvals), "k-", label="Fit")

            # print(fitted_line)

            if avfrac is not None:
                # fit a line with hyperfit to account for correlated uncertainties
                ndata = np.sum(gvals)
                hfdata, hfcov = np.zeros((2, ndata)), np.zeros((2, 2, ndata))
                corr_xy = -1.0 * avfrac

                hfdata[0, :] = xvals[gvals]
                hfdata[1, :] = yvals[gvals]
                for k in range(ndata):
                    hfcov[0, 0, k] = xvals_unc[gvals][k] ** 2
                    hfcov[0, 1, k] = (
                        xvals_unc[gvals][k] * yvals_unc[gvals][k] * corr_xy[gvals][k]
                    )
                    hfcov[1, 0, k] = (
                        xvals_unc[gvals][k] * yvals_unc[gvals][k] * corr_xy[gvals][k]
                    )
                    hfcov[1, 1, k] = yvals_unc[gvals][k] ** 2

                hf_fit = HFLinFit(hfdata, hfcov)

                ds = 0.5 * np.absolute(fitted_line.slope)
                di = 0.5 * np.absolute(fitted_line.intercept)
                bounds = (
                    (fitted_line.slope - ds, fitted_line.slope + ds),
                    (fitted_line.intercept - di, fitted_line.intercept + di),
                    (1.0e-5, 5.0),
                )
                hf_fit_params = hf_fit.optimize(bounds, verbose=False)
                mod_yvals_hf = hf_fit.coords[1] + hf_fit.coords[0] * mod_xvals
                ax[i].plot(mod_xvals, mod_yvals_hf, "k--", label="HF Fit")

                ax[i].plot(mod_xvals, mod_yvals_hf - hf_fit.vert_scat, "k:")
                ax[i].plot(mod_xvals, mod_yvals_hf + hf_fit.vert_scat, "k:")

                # print(hf_fit_params)

                sigmas = hf_fit.get_sigmas()

                # Generate ellipses
                for k in range(ndata):
                    e = Ellipse(
                        xy=[hfdata[0][k], hfdata[1][k]],
                        width=2 * xvals_unc[gvals][k],
                        height=2 * yvals_unc[gvals][k],
                        # angle=0.0,
                        alpha=0.20,
                        color=cm.viridis(sigmas[k] / np.amax(sigmas))[0],
                        # edgecolor="k",
                        angle=90.0 - np.rad2deg(np.arccos(corr_xy[k])),
                        # angle=np.rad2deg(corr_xy[k] * 45.0 * np.pi / 180.0),
                    )
                    # print(yvals[gvals][k], yvals_unc[gvals][k], np.rad2deg(corr_xy[k] * 45.0 * np.pi / 180.0))
                    ax[i].add_artist(e)

    fax[0, 1].set_xlim(xrange)

    # for 2nd x-axis with R(V) values
    new_ticks = np.array([1.0 / 2.0, 1.0 / 3.0, 1.0 / 4.0, 1.0 / 5.0, 1.0 / 6.0])
    new_ticks_labels = ["%.1f" % z for z in 1.0 / new_ticks]

    for i in range(4):
        fax[1, i].set_xlabel(labx)

        if not args.rv:
            # add 2nd x-axis with R(V) values
            tax = fax[0, i].twiny()
            tax.set_xlim(fax[0, i].get_xlim())
            tax.set_xticks(new_ticks)
            tax.set_xticklabels(new_ticks_labels)
            tax.set_xlabel(r"$R(V)$")

    for i in range(2):
        fax[i, 0].set_ylabel(laby)

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
