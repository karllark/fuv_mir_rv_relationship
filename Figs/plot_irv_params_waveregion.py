import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

from astropy.table import QTable
import astropy.units as u
from astropy.stats import sigma_clip
from astropy.modeling.fitting import LevMarLSQFitter, FittingWithOutlierRemoval
from astropy.modeling.models import Drude1D, Polynomial1D  # , Legendre1D

from dust_extinction.shapes import FM90

from helpers import G21mod, G22  # , G22pow  # , G22opt


def plot_irv_ssamp(
    ax, itab, label, color="k", linestyle="solid", simpfit=False, inst=None
):

    # remove bad regions
    bregions = (
        np.array(
            [
                [1190.0, 1235.0],
                [1370.0, 1408.0],
                [1515.0, 1563.0],
                [6560.0, 6570.0],
                [41000.0, 50000.0],
            ]
        )
        * u.AA
    )
    for cbad in bregions:
        bvals = np.logical_and(itab["waves"] > cbad[0], itab["waves"] < cbad[1])
        itab["npts"][bvals] = 0

    # find regions that have large uncertainties in the intercept and remove
    # gindxs = np.where(itab["npts"] > 0)
    # bvals = (itab["hfintercepts"][gindxs] / itab["hfintercepts_std"][gindxs]) < 10
    # itab["npts"][gindxs][bvals] = 0
    # print(itab["waves"][gindxs][bvals])

    # trim ends
    if inst == "IUE":
        bvals = itab["waves"] > 0.3 * u.micron
        itab["npts"][bvals] = 0
    elif inst == "STIS":
        bvals = itab["waves"] > 0.95 * u.micron
        itab["npts"][bvals] = 0
    elif inst == "SpeXLXD":
        bvals = itab["waves"] > 5.5 * u.micron
        itab["npts"][bvals] = 0

    # set to NAN so they are not plotted
    bvals = itab["npts"] == 0
    itab["hfslopes"][bvals] = np.NAN
    itab["hfintercepts"][bvals] = np.NAN
    itab["hfsigmas"][bvals] = np.NAN
    itab["hfrmss"][bvals] = np.NAN
    gvals = itab["npts"] >= 0
    if simpfit:
        for k, cname in enumerate(["intercepts", "slopes", "rmss"]):
            ax[k * 2].plot(
                itab["waves"][gvals],
                itab[cname][gvals],
                linestyle="dashed",
                color=color,
                alpha=0.75,
            )
    if "hfslopes" in itab.colnames:
        for k, cname in enumerate(["hfintercepts", "hfslopes", "hfsigmas"]):
            ax[k * 2].plot(
                itab["waves"][gvals],
                itab[cname][gvals],
                linestyle=linestyle,
                color=color,
                label=label,
                alpha=0.75,
            )
        if "hfslopes_std" in itab.colnames:
            for k, cname in enumerate(["hfintercepts", "hfslopes", "hfsigmas"]):
                ax[k * 2].fill_between(
                    itab["waves"][gvals].value,
                    itab[cname][gvals] - itab[f"{cname}_std"],
                    itab[cname][gvals] + itab[f"{cname}_std"],
                    color=color,
                    alpha=0.25,
                )

    return (
        itab["npts"],
        itab["waves"],
        itab["hfintercepts"],
        itab["hfslopes"],
        itab["hfintercepts_std"],
        itab["hfslopes_std"],
    )


def plot_resid(ax, data, dindx, model, color):
    """
    Plot the residuals to the model
    """
    bvals = data[0] <= 0
    data[dindx][bvals] = np.NAN

    # only plot where the model is valid
    gvals = (data[1].value >= 1.0 / model.x_range[1]) & (
        data[1].value <= 1.0 / model.x_range[0]
    )
    fitx = 1.0 / data[1][gvals].value
    ax.plot(
        data[1][gvals],
        # (data[dindx][gvals] - model(fitx)) / model(fitx),
        data[dindx][gvals] - model(fitx),
        color=color,
        alpha=0.75,
    )


def plot_wavereg(ax, models, datasets, colors, wrange, no_weights=False):
    """
    Do the fits and plot the fits and residuals
    """
    npts = []
    waves = []
    intercepts = []
    intercepts_unc = []
    slopes = []
    slopes_unc = []
    for cdata in datasets:
        npts.append(cdata[0])
        waves.append(cdata[1])
        intercepts.append(cdata[2])
        slopes.append(cdata[3])
        intercepts_unc.append(cdata[4])
        slopes_unc.append(cdata[5])
    all_npts = np.concatenate(npts)
    all_waves = np.concatenate(waves)
    all_intercepts = np.concatenate(intercepts)
    all_intercepts_unc = np.concatenate(intercepts_unc)
    all_slopes = np.concatenate(slopes)
    all_slopes_unc = np.concatenate(slopes_unc)

    if no_weights:
        tnpts = len(all_waves)
        all_intercepts_unc = np.full(tnpts, 1.0)
        all_slopes_unc = np.full(tnpts, 1.0)

    sindxs = np.argsort(all_waves)
    all_waves = all_waves[sindxs]
    all_npts = all_npts[sindxs]
    all_intercepts = all_intercepts[sindxs]
    all_slopes = all_slopes[sindxs]
    all_intercepts_unc = all_intercepts_unc[sindxs]
    all_slopes_unc = all_slopes_unc[sindxs]

    gvals = (all_npts > 0) & (all_waves >= wrange[0]) & (all_waves <= wrange[1])

    fit = LevMarLSQFitter()
    or_fit = FittingWithOutlierRemoval(fit, sigma_clip, niter=3, sigma=5.0)
    rejsym = "kx"

    # intercept
    # cmodelfit = fit(
    #     cmodelinit, 1.0 / all_waves[gvals].value, all_intercepts[gvals], maxiter=500
    # )
    fitx = 1.0 / all_waves[gvals].value
    cmodelfit, mask = or_fit(
        models[0], fitx, all_intercepts[gvals], weights=1.0 / all_intercepts_unc[gvals]
    )
    filtered_data = np.ma.masked_array(all_intercepts[gvals], mask=~mask)
    fitted_models = [cmodelfit]

    print("intercepts")
    print(cmodelfit.param_names)
    print(cmodelfit.parameters)

    ax[0].plot(all_waves[gvals], cmodelfit(fitx))
    ax[0].plot(all_waves[gvals], filtered_data, rejsym, label="rejected")

    for cdata, ccolor in zip(datasets, colors):
        plot_resid(ax[1], cdata, 2, cmodelfit, ccolor)
    filtered_data2 = np.ma.masked_array(
        all_intercepts[gvals] - cmodelfit(fitx), mask=~mask
    )
    ax[1].plot(all_waves[gvals], filtered_data2, rejsym, label="rejected")

    # slope
    cmodelfit, mask = or_fit(
        models[1], fitx, all_slopes[gvals], weights=1.0 / all_slopes_unc[gvals]
    )
    filtered_data = np.ma.masked_array(all_slopes[gvals], mask=~mask)
    fitted_models.append(cmodelfit)

    print("slopes")
    print(cmodelfit.param_names)
    print(cmodelfit.parameters)

    ax[2].plot(all_waves[gvals], cmodelfit(fitx))
    ax[2].plot(all_waves[gvals], filtered_data, rejsym, label="rejected")

    for cdata, ccolor in zip(datasets, colors):
        plot_resid(ax[3], cdata, 3, cmodelfit, ccolor)
    filtered_data2 = np.ma.masked_array(all_slopes[gvals] - cmodelfit(fitx), mask=~mask)
    print("total masked", np.sum(mask))
    ax[3].plot(all_waves[gvals], filtered_data2, rejsym, label="rejected")

    return fitted_models


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wavereg",
        choices=["uv", "opt", "ir", "all"],
        default="all",
        help="Wavelength region to plot",
    )
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    # get irv parameters
    gor09_fuse = QTable.read("results/gor09_fuse_irv_params.fits")
    gor09_iue = QTable.read("results/gor09_iue_irv_params.fits")

    fit19_stis = QTable.read("results/fit19_stis_irv_params.fits")

    gor21_iue = QTable.read("results/gor21_iue_irv_params.fits")
    gor21_irs = QTable.read("results/gor21_irs_irv_params.fits")

    dec22_iue = QTable.read("results/dec22_iue_irv_params.fits")
    dec22_spexsxd = QTable.read("results/dec22_spexsxd_irv_params.fits")
    dec22_spexlxd = QTable.read("results/dec22_spexlxd_irv_params.fits")

    # adjust to match nir
    # fit19_stis["hfintercepts"] -= 0.04

    # remove UV from Fitzpatrick19 (only want the optical STIS data)
    gvals = fit19_stis["waves"] > 0.30 * u.micron
    fit19_stis = fit19_stis[gvals]

    aiue_iue = QTable.read("results/aiue_iue_irv_params.fits")

    # setup plot
    fontsize = 14
    font = {"size": fontsize}
    plt.rc("font", **font)
    plt.rc("lines", linewidth=1.5)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("ytick.major", width=2)
    fig, ax = plt.subplots(
        nrows=5,
        ncols=1,
        figsize=(10, 9),
        sharex="col",
        gridspec_kw={
            "width_ratios": [1],
            "height_ratios": [3, 1, 3, 1, 3],
            "wspace": 0.01,
            "hspace": 0.01,
        },
        constrained_layout=True,
    )

    gor09_color = "blue"
    fit19_color = "green"
    dec22_color = "red"
    dec22_color = "red"
    gor21_color = "purple"
    aiue_color = "orange"

    # plot parameters
    yrange_b_type = "linear"
    yrange_s_type = "linear"
    if args.wavereg == "uv":
        gor09_res1 = plot_irv_ssamp(ax, gor09_fuse, "G09", color=gor09_color)
        alliue_res = plot_irv_ssamp(ax, aiue_iue, "All", color=aiue_color, inst="IUE")
        fit19_res = plot_irv_ssamp(
            ax, fit19_stis, "F19", color=fit19_color, inst="STIS"
        )
        xrange = [0.09, 0.33]
        yrange_a_type = "linear"
        yrange_a = [1.0, 7.5]
        yrange_b = [1.0, 50.0]
        yrange_s = [0.0, 1.5]
        xticks = [0.09, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3]

        # increase the weight of the 2175 A bump region to ensure it is fit well
        # as has been done since FM90
        # done by decreasing the uncdertainties
        bvals = (alliue_res[1] > 0.20 * u.micron) & (alliue_res[1] < 0.24 * u.micron)
        alliue_res[4][bvals] /= 5.0
        alliue_res[5][bvals] /= 5.0

        # decrease the weight of the IUE data in the "poor" flux calibration/stability
        # region - see section 4.2 of Fitzpatrick et al. 2019
        bvals = alliue_res[1] > 0.27 * u.micron
        alliue_res[4][bvals] *= 5.0
        alliue_res[5][bvals] *= 5.0

        # fitting
        datasets = [alliue_res, gor09_res1, fit19_res]
        colors = [aiue_color, gor09_color, fit19_color]
        fm90 = FM90()
        plot_wavereg(ax, [fm90, fm90], datasets, colors, wrange=[0.09, 0.33] * u.micron)
        ax[1].set_ylim(-0.2, 0.2)
        ax[3].set_ylim(-3.0, 3.0)
    elif args.wavereg == "opt":
        # alliue_res = plot_irv_ssamp(ax, aiue_iue, "All", color=aiue_color, inst="IUE")
        fit19_res = plot_irv_ssamp(
            ax, fit19_stis, "F19", color=fit19_color, inst="STIS"
        )
        dec22_res1 = plot_irv_ssamp(ax, dec22_spexsxd, "D22", color=dec22_color)
        xrange = [0.30, 1.1]
        yrange_a_type = "linear"
        yrange_a = [0.2, 2.2]
        yrange_b = [-1.5, 6.0]
        yrange_s = [0.0, 0.08]
        xticks = [0.3, 0.35, 0.45, 0.55, 0.7, 0.9, 1.0]

        # fitting
        datasets = [fit19_res, dec22_res1]
        colors = [fit19_color, dec22_color]
        # g22opt = G22opt()
        g22opt = (
            Polynomial1D(5)
            # Legendre1D(3)
            + Drude1D(amplitude=0.1, x_0=2.238, fwhm=0.243)
            + Drude1D(amplitude=0.1, x_0=2.054, fwhm=0.179)
            + Drude1D(amplitude=0.1, x_0=1.587, fwhm=0.243)
        )
        # g22opt[1].amplitude.bounds = [0.0, 0.2]
        # g22opt[2].amplitude.bounds = [0.0, 0.2]
        g22opt[1].fwhm.fixed = True
        g22opt[2].fwhm.fixed = True
        g22opt[3].fwhm.fixed = True
        g22opt[1].x_0.fixed = True
        g22opt[2].x_0.fixed = True
        g22opt[3].x_0.fixed = True
        g22opt.x_range = [1.0 / 1.1, 1.0 / 0.27]

        # do not use weights in the fitting
        # systematic issues with the stellar atmosphere models for the Paschen series
        # idea is to equally weight the F19 and D22 results as both have
        # isues in this wavelength range and the F19 sample has ~5x more stars
        fitted_models = plot_wavereg(
            ax,
            [g22opt, g22opt],
            datasets,
            colors,
            wrange=[0.30, 1.1] * u.micron,
            no_weights=True,
        )
        ax[1].set_ylim(-0.03, 0.1)
        ax[3].set_ylim(-0.3, 0.3)

        # plotting the components
        modx = np.linspace(0.30, 1.0, 100)
        ax[0].plot(modx, fitted_models[0][0](1.0 / modx), "k:")

        gvals = (datasets[0][1].value >= 1.0 / fitted_models[0].x_range[1]) & (
            datasets[0][1].value <= 1.0 / fitted_models[0].x_range[0]
        )
        fitx = 1.0 / datasets[0][1][gvals].value
        ax[1].plot(
            datasets[0][1][gvals].value,
            datasets[0][2][gvals] - fitted_models[0][0](fitx),
            "k--",
        )
        ax[1].plot(
            datasets[0][1][gvals].value,
            fitted_models[0][1](fitx),
            "k:",
        )
        ax[1].plot(
            datasets[0][1][gvals].value,
            fitted_models[0][2](fitx),
            "k:",
        )
        ax[1].plot(
            datasets[0][1][gvals].value,
            fitted_models[0][3](fitx),
            "k:",
        )

        ax[2].plot(modx, fitted_models[1][0](1.0 / modx), "k:")

    elif args.wavereg == "ir":
        # fit19_res = plot_irv_ssamp(
        #     ax, fit19_stis, "F19", color=fit19_color, inst="STIS"
        # )
        dec22_res1 = plot_irv_ssamp(ax, dec22_spexsxd, "D22", color=dec22_color)
        dec22_res2 = plot_irv_ssamp(
            ax,
            dec22_spexlxd,
            None,
            color=dec22_color,
            inst="SpeXLXD",
        )
        gor21_res = plot_irv_ssamp(ax, gor21_irs, "G21", color=gor21_color)
        xrange = [0.8, 35.0]
        yrange_a_type = "log"
        yrange_a = [0.02, 0.6]
        yrange_b = [-1.0, 0.5]
        yrange_s = [0.0, 0.08]
        xticks = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]

        # fitting
        datasets = [dec22_res1, dec22_res2, gor21_res]
        colors = [dec22_color, dec22_color, gor21_color]
        g21mod = G21mod()
        # g21mod.ice_amp.fixed = True
        g21mod.swave.bounds = [3.0, 5.0]
        g21mod.ice_fwhm.fixed = True
        g21mod.ice_center.fixed = True
        g21mod.ice_asym.fixed = True
        # g21mod.sil1_asym = -0.6
        # g21mod.sil1_asym.fixed = True

        # irpow = G22pow()
        irpow = Polynomial1D(6)
        irpow.x_range = [1.0 / 40.0, 1.0 / 0.8]

        plot_wavereg(
            ax,
            [g21mod, irpow],
            datasets,
            colors,
            wrange=[1.0, 40.0] * u.micron,
            no_weights=False,
        )
        ax[1].set_ylim(-0.015, 0.015)
        ax[3].set_ylim(-0.2, 0.2)
    else:
        gor09_res1 = plot_irv_ssamp(ax, gor09_fuse, "G09", color=gor09_color)
        alliue_res = plot_irv_ssamp(ax, aiue_iue, "All", color=aiue_color, inst="IUE")
        fit19_res = plot_irv_ssamp(
            ax, fit19_stis, "F19", color=fit19_color, inst="STIS"
        )
        dec22_res1 = plot_irv_ssamp(ax, dec22_spexsxd, "D22", color=dec22_color)
        dec22_res2 = plot_irv_ssamp(
            ax,
            dec22_spexlxd,
            None,
            color=dec22_color,
            inst="SpeXLXD",
        )
        gor21_res = plot_irv_ssamp(ax, gor21_irs, "G21", color=gor21_color)
        xrange = [0.09, 35.0]
        yrange_a_type = "log"
        yrange_a = [0.02, 7.5]
        yrange_b_type = "symlog"
        yrange_b = [-2.0, 50.0]
        yrange_s_type = "log"
        yrange_s = [0.001, 1.0]
        xticks = [
            0.1,
            0.2,
            0.3,
            0.5,
            0.8,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            5.0,
            7.0,
            10.0,
            15.0,
            20.0,
            30.0,
        ]

        g22mod = G22(Rv=3.1)
        modx = np.logspace(np.log10(0.091), np.log10(34.0), 1000) * u.micron
        g22rv31 = g22mod(modx)
        ax[0].plot(modx, g22mod.a, "k-", alpha=0.5)
        ax[2].plot(modx, g22mod.b, "k-", alpha=0.5)

    # set the wavelength range for all the plots
    ax[4].set_xscale("log")
    ax[4].set_xlim(xrange)
    ax[4].set_xlabel(r"$\lambda$ [$\mu$m]")

    ax[0].set_yscale(yrange_a_type)
    ax[2].set_yscale(yrange_b_type)
    ax[4].set_yscale(yrange_s_type)
    ax[0].set_ylim(yrange_a)
    ax[2].set_ylim(yrange_b)
    ax[4].set_ylim(yrange_s)
    ax[0].set_ylabel("intercept (a)")
    ax[1].set_ylabel("a - fit")
    ax[2].set_ylabel("slope (b)")
    ax[3].set_ylabel("b - fit")
    ax[4].set_ylabel(r"scatter ($\sigma$)")
    # ax[3, 0].set_ylabel("scatter")

    for i in range(5):
        ax[i].axhline(linestyle="--", alpha=0.25, color="k", linewidth=2)

    ax[0].legend(ncol=2)

    ax[4].xaxis.set_major_formatter(ScalarFormatter())
    ax[4].xaxis.set_minor_formatter(ScalarFormatter())
    ax[4].set_xticks(xticks, minor=True)

    fname = f"fuv_mir_rv_fit_params_{args.wavereg}"
    if args.png:
        fig.savefig(f"{fname}.png")
    elif args.pdf:
        fig.savefig(f"{fname}.pdf")
    else:
        plt.show()
