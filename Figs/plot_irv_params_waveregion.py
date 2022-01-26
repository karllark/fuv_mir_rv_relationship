import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

from astropy.table import QTable
import astropy.units as u
from astropy.stats import sigma_clip
from astropy.modeling.fitting import LevMarLSQFitter, FittingWithOutlierRemoval
from astropy.modeling.models import Drude1D, Polynomial1D, Legendre1D

from dust_extinction.shapes import FM90

from helpers import G21mod, G22opt, G22pow


def plot_irv_ssamp(
    ax, itab, label, color="k", linestyle="solid", simpfit=False, inst=None
):

    # remove bad regions
    bregions = np.array([[1190.0, 1235.0], [1370.0, 1408.0], [1515.0, 1563.0]]) * u.AA
    for cbad in bregions:
        bvals = np.logical_and(itab["waves"] > cbad[0], itab["waves"] < cbad[1])
        itab["npts"][bvals] = 0

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
        ax[1].plot(
            itab["waves"][gvals],
            itab["slopes"][gvals],
            linestyle="dashed",
            color=color,
            alpha=0.75,
        )
        ax[0].plot(
            itab["waves"][gvals],
            itab["intercepts"][gvals],
            linestyle="dashed",
            color=color,
            alpha=0.75,
        )
        if "rmss" in itab.colnames:
            ax[2].plot(
                itab["waves"][gvals],
                itab["rmss"][gvals],
                linestyle="dashed",
                color=color,
                alpha=0.75,
            )
        # else:
        #     ax[3, i].plot(
        #         itab["waves"][gvals],
        #         itab["sigmas"][gvals],
        #         linestyle="dashed",
        #         color=color,
        #         alpha=0.75,
        #     )
    if "hfslopes" in itab.colnames:
        ax[0].plot(
            itab["waves"][gvals],
            itab["hfintercepts"][gvals],
            linestyle=linestyle,
            color=color,
            label=label,
            alpha=0.75,
        )
        ax[2].plot(
            itab["waves"][gvals],
            itab["hfslopes"][gvals],
            linestyle=linestyle,
            color=color,
            label=label,
            alpha=0.75,
        )
        ax[4].plot(
            itab["waves"][gvals],
            itab["hfsigmas"][gvals],
            linestyle=linestyle,
            color=color,
            label=label,
            alpha=0.75,
        )
        if "hfslopes_std" in itab.colnames:
            ax[0].plot(
                itab["waves"][gvals],
                itab["hfintercepts"][gvals] + itab["hfintercepts_std"][gvals],
                linestyle="dashed",
                color=color,
                label=label,
                alpha=0.75,
            )
            ax[0].plot(
                itab["waves"][gvals],
                itab["hfintercepts"][gvals] - itab["hfintercepts_std"][gvals],
                linestyle="dashed",
                color=color,
                label=label,
                alpha=0.75,
            )
            ax[2].plot(
                itab["waves"][gvals],
                itab["hfslopes"][gvals] + itab["hfslopes_std"],
                linestyle="dashed",
                color=color,
                label=label,
                alpha=0.75,
            )
            ax[2].plot(
                itab["waves"][gvals],
                itab["hfslopes"][gvals] - itab["hfslopes_std"],
                linestyle="dashed",
                color=color,
                label=label,
                alpha=0.75,
            )

    return (itab["npts"], itab["waves"], itab["hfintercepts"], itab["hfslopes"])


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
        data[dindx][gvals] - model(fitx),
        color=color,
        alpha=0.75,
    )


def plot_wavereg(ax, models, datasets, colors, wrange):
    """
    Do the fits and plot the fits and residuals
    """
    npts = []
    waves = []
    intercepts = []
    slopes = []
    for cdata in datasets:
        npts.append(cdata[0])
        waves.append(cdata[1])
        intercepts.append(cdata[2])
        slopes.append(cdata[3])
    all_npts = np.concatenate(npts)
    all_waves = np.concatenate(waves)
    all_intercepts = np.concatenate(intercepts)
    all_slopes = np.concatenate(slopes)

    sindxs = np.argsort(all_waves)
    all_waves = all_waves[sindxs]
    all_npts = all_npts[sindxs]
    all_intercepts = all_intercepts[sindxs]
    all_slopes = all_slopes[sindxs]

    gvals = (all_npts > 0) & (all_waves >= wrange[0]) & (all_waves <= wrange[1])

    fit = LevMarLSQFitter()
    or_fit = FittingWithOutlierRemoval(fit, sigma_clip, niter=3, sigma=3.0)
    rejsym = "kx"

    # intercept
    # cmodelfit = fit(
    #     cmodelinit, 1.0 / all_waves[gvals].value, all_intercepts[gvals], maxiter=500
    # )
    fitx = 1.0 / all_waves[gvals].value
    cmodelfit, mask = or_fit(models[0], fitx, all_intercepts[gvals])
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
    cmodelfit, mask = or_fit(models[1], fitx, all_slopes[gvals])
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
        choices=["uv", "opt", "ir"],
        default="ir",
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
    aiue_color = "black"

    # plot parameters
    if args.wavereg == "uv":
        gor09_res1 = plot_irv_ssamp(ax, gor09_fuse, "G09", color=gor09_color)
        alliue_res = plot_irv_ssamp(ax, aiue_iue, "All", color=aiue_color, inst="IUE")
        xrange = [0.09, 0.32]
        yrange_a_type = "linear"
        yrange_a = [1.0, 7.5]
        yrange_b = [1.0, 50.0]
        yrange_s = [0.0, 2.0]
        xticks = [0.09, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3]

        # fitting
        datasets = [alliue_res, gor09_res1]
        colors = [aiue_color, gor09_color]
        fm90 = FM90()
        plot_wavereg(ax, [fm90, fm90], datasets, colors, wrange=[0.09, 0.35] * u.micron)
        ax[1].set_ylim(-0.2, 0.2)
        ax[3].set_ylim(-3.0, 3.0)
    elif args.wavereg == "opt":
        fit19_res = plot_irv_ssamp(
            ax, fit19_stis, "F19", color=fit19_color, inst="STIS"
        )
        dec22_res1 = plot_irv_ssamp(ax, dec22_spexsxd, "D22", color=dec22_color)
        xrange = [0.30, 1.0]
        yrange_a_type = "linear"
        yrange_a = [0.2, 2.0]
        yrange_b = [-1.5, 3.5]
        yrange_s = [0.0, 0.08]
        xticks = [0.3, 0.35, 0.45, 0.55, 0.7, 0.9, 1.0]

        # fitting
        datasets = [fit19_res, dec22_res1]
        colors = [fit19_color, dec22_color]
        # g22opt = G22opt()
        g22opt = (
            Polynomial1D(4)
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
        # g22opt[1].x_0.fixed = True
        # g22opt[2].x_0.fixed = True
        # g22opt[3].x_0.fixed = True
        g22opt.x_range = [1.0 / 1.0, 1.0 / 0.3]
        fitted_models = plot_wavereg(
            ax, [g22opt, g22opt], datasets, colors, wrange=[0.30, 1.0] * u.micron
        )
        # ax[1].set_ylim(-0.2, 0.2)
        ax[3].set_ylim(-0.2, 0.2)

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
        # ax[1].plot(
        #     datasets[0][1][gvals].value,
        #     fitted_models[0][3](fitx),
        #     "k:",
        # )

        ax[2].plot(modx, fitted_models[1][0](1.0 / modx), "k:")

    else:
        dec22_res1 = plot_irv_ssamp(ax, dec22_spexsxd, "D22", color=dec22_color)
        dec22_res2 = plot_irv_ssamp(
            ax,
            dec22_spexlxd,
            None,
            color=dec22_color,
            inst="SpeXLXD",
        )
        gor21_res = plot_irv_ssamp(ax, gor21_irs, "G21", color=gor21_color)
        xrange = [1.0, 35.0]
        yrange_a_type = "log"
        yrange_a = [0.02, 0.4]
        yrange_b = [-1.0, 0.5]
        yrange_s = [0.0, 0.08]
        xticks = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]

        # fitting
        datasets = [dec22_res1, dec22_res2, gor21_res]
        colors = [dec22_color, dec22_color, gor21_color]
        g21mod = G21mod()
        g21mod.ice_amp.fixed = True
        g21mod.ice_fwhm.fixed = True
        g21mod.ice_center.fixed = True
        g21mod.ice_asym.fixed = True

        irpow = G22pow()
        plot_wavereg(
            ax, [g21mod, irpow], datasets, colors, wrange=[1.0, 40.0] * u.micron
        )
        ax[1].set_ylim(-0.015, 0.015)
        # ax[3].set_ylim(-0.5, 0.5)

    # set the wavelength range for all the plots
    ax[4].set_xscale("log")
    ax[4].set_xlim(xrange)
    ax[4].set_xlabel(r"$\lambda$ [$\mu$m]")

    ax[0].set_yscale(yrange_a_type)
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
