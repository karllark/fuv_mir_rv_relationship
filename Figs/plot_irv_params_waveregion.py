import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

from astropy.table import QTable
import astropy.units as u
from astropy.stats import sigma_clip
from astropy.modeling.fitting import LevMarLSQFitter, FittingWithOutlierRemoval

from dust_extinction.shapes import FM90

from plot_irv_params import G21mod


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
        # ax[3].plot(
        #     itab["waves"][gvals],
        #     itab["hfrmss"][gvals],
        #     color=color,
        #     label=label,
        #     alpha=0.75,
        # )

    return (itab["npts"], itab["waves"], itab["hfintercepts"], itab["hfslopes"])


def plot_resid(ax, data, dindx, model, color):
    """
    Plot the residuals to the model
    """
    bvals = data[0] <= 0
    data[dindx][bvals] = np.NAN

    ax.plot(
        data[1],
        data[dindx] - model(data[1]),
        color=color,
        alpha=0.75,
    )


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
        # gor09_res2 = plot_irv_ssamp(
        #     ax, gor09_iue, None, color=gor09_color, inst="IUE", linestyle="dashed"
        # )
        # gor21_res = plot_irv_ssamp(
        #     ax, gor21_iue, "G21", color=gor21_color, inst="IUE",
        # )
        # dec22_res = plot_irv_ssamp(
        #     ax, dec22_iue, "D22", color=dec22_color, inst="IUE"
        # )
        # fit19_res = plot_irv_ssamp(
        #     ax, fit19_stis, "F19", color=fit19_color, inst="STIS"
        # )
        alliue_res = plot_irv_ssamp(ax, aiue_iue, "All", color=aiue_color, inst="IUE")
        xrange = [0.09, 0.32]
        yrange_a_type = "linear"
        yrange_a = [1.0, 7.5]
        yrange_b = [1.0, 50.0]
        yrange_s = [0.0, 2.0]
        xticks = [0.09, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3]
    elif args.wavereg == "opt":
        fit19_res = plot_irv_ssamp(
            ax, fit19_stis, "F19", color=fit19_color, inst="STIS"
        )
        dec22_res1 = plot_irv_ssamp(ax, dec22_spexsxd, "D22", color=dec22_color)
        xrange = [0.30, 1.0]
        yrange_a_type = "linear"
        yrange_a = [0.2, 2.0]
        yrange_b = [-1.0, 3.5]
        yrange_s = [0.0, 0.08]
        xticks = [0.3, 0.35, 0.45, 0.55, 0.7, 0.9, 1.0]
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

    # do the fitting
    fit = LevMarLSQFitter()
    or_fit = FittingWithOutlierRemoval(fit, sigma_clip, niter=3, sigma=3.0)
    rejsym = "kx"

    if args.wavereg == "uv":
        # fit FM90
        all_npts = np.concatenate((alliue_res[0], gor09_res1[0]))
        all_waves = np.concatenate((alliue_res[1], gor09_res1[1]))
        all_intercepts = np.concatenate((alliue_res[2], gor09_res1[2]))
        all_slopes = np.concatenate((alliue_res[3], gor09_res1[3]))

        # all_npts = alliue_res[0]
        # all_waves = alliue_res[1]
        # all_intercepts = alliue_res[2]
        # all_slopes = alliue_res[3]

        sindxs = np.argsort(all_waves)
        all_waves = all_waves[sindxs]
        all_npts = all_npts[sindxs]
        all_intercepts = all_intercepts[sindxs]
        all_slopes = all_slopes[sindxs]

        gvals = all_npts > 0

        fm90init = FM90()

        # intercept
        # fm90fit = fit(
        #     fm90init, 1.0 / all_waves[gvals].value, all_intercepts[gvals], maxiter=500
        # )
        #
        # print("FM90 intercepts")
        # print(fm90fit.param_names)
        # print(fm90fit.parameters)

        # fm90fit, mask = or_fit(fm90init, all_waves[gvals], all_intercepts[gvals])
        fm90fit, mask = or_fit(fm90init, 1.0 / all_waves[gvals].value, all_intercepts[gvals])
        filtered_data = np.ma.masked_array(all_intercepts[gvals], mask=~mask)

        print("FM90 intercepts")
        print(fm90fit.param_names)
        print(fm90fit.parameters)

        ax[0].plot(all_waves[gvals], fm90fit(all_waves[gvals]))
        ax[0].plot(all_waves[gvals], filtered_data, rejsym, label="rejected data")

        plot_resid(ax[1], gor09_res1, 2, fm90fit, gor09_color)
        plot_resid(ax[1], alliue_res, 2, fm90fit, aiue_color)
        ax[1].set_ylim(-0.2, 0.2)

        # slope
        # fm90fit = fit(
        #     fm90init, 1.0 / all_waves[gvals].value, all_slopes[gvals], maxiter=500
        # )
        # print("FM90 slopes")
        # print(fm90fit.param_names)
        # print(fm90fit.parameters)

        fm90fit, mask = or_fit(fm90init, 1.0 / all_waves[gvals].value, all_slopes[gvals])
        filtered_data = np.ma.masked_array(all_slopes[gvals], mask=~mask)

        print("FM90 intercepts")
        print(fm90fit.param_names)
        print(fm90fit.parameters)

        ax[2].plot(all_waves[gvals], fm90fit(all_waves[gvals]))
        ax[2].plot(all_waves[gvals], filtered_data, rejsym, label="rejected data")

        plot_resid(ax[3], gor09_res1, 3, fm90fit, gor09_color)
        plot_resid(ax[3], alliue_res, 3, fm90fit, aiue_color)
        ax[3].set_ylim(-3.0, 3.0)

    elif args.wavereg == "ir":
        # fit G21 modified to > 1 micron data
        all_npts = np.concatenate((dec22_res1[0], dec22_res2[0], gor21_res[0]))
        all_waves = np.concatenate((dec22_res1[1], dec22_res2[1], gor21_res[1]))
        all_intercepts = np.concatenate((dec22_res1[2], dec22_res2[2], gor21_res[2]))
        all_slopes = np.concatenate((dec22_res1[3], dec22_res2[3], gor21_res[3]))

        # fit intercepts
        g21init = G21mod()
        g21init.ice_amp.fixed = True
        g21init.ice_fwhm.fixed = True
        g21init.ice_center.fixed = True
        g21init.ice_asym.fixed = True
        # g21init.sil1_asym.fixed = True
        bvals = all_waves < 1.0 * u.micron
        all_npts[bvals] = 0
        gvals = all_npts > 0
        # g21fit = fit(
        #     g21init, 1.0 / all_waves[gvals].value, all_intercepts[gvals], maxiter=500
        # )
        g21fit, mask = or_fit(g21init, 1.0 / all_waves[gvals].value, all_intercepts[gvals])
        filtered_data = np.ma.masked_array(all_intercepts[gvals], mask=~mask)
        print(g21fit.param_names)
        print(g21fit.parameters)
        print(g21init.parameters)
        print(fit.fit_info["message"])
        ax[0].plot(all_waves[gvals], g21fit(all_waves[gvals]))
        ax[0].plot(all_waves[gvals], filtered_data, rejsym, label="rejected data")

        plot_resid(ax[1], dec22_res1, 2, g21fit, dec22_color)
        plot_resid(ax[1], dec22_res2, 2, g21fit, dec22_color)
        plot_resid(ax[1], gor21_res, 2, g21fit, gor21_color)
        ax[1].set_ylim(-0.015, 0.015)

        # fit slopes
        g21init = G21mod()
        g21init.scale = -0.8
        g21init.scale.bounds = [-2.0, 0.0]
        g21init.alpha2 = 0.0
        g21init.alpha2.fixed = True
        # g21init.ice_amp = 0.0
        # g21init.ice_amp.fixed = True
        g21init.ice_fwhm.fixed = True
        g21init.ice_center.fixed = True
        g21init.ice_asym.fixed = True
        g21init.sil1_amp = 0.0
        g21init.sil1_amp.fixed = True
        g21init.sil2_amp = 0.0
        g21init.sil2_amp.fixed = True
        fit = LevMarLSQFitter()
        # g21fit = fit(
        #     g21init, 1.0 / all_waves[gvals].value, all_slopes[gvals], maxiter=500
        # )
        g21fit, mask = or_fit(g21init, 1.0 / all_waves[gvals].value, all_slopes[gvals])
        filtered_data = np.ma.masked_array(all_slopes[gvals], mask=~mask)
        print(g21fit.param_names)
        print(g21fit.parameters)
        print(g21init.parameters)
        print(fit.fit_info["message"])
        ax[2].plot(all_waves[gvals], g21fit(all_waves[gvals]))
        ax[2].plot(all_waves[gvals], filtered_data, rejsym, label="rejected data")

        plot_resid(ax[3], dec22_res1, 3, g21fit, dec22_color)
        plot_resid(ax[3], dec22_res2, 3, g21fit, dec22_color)
        plot_resid(ax[3], gor21_res, 3, g21fit, gor21_color)
        filtered_data2 = np.ma.masked_array(all_slopes[gvals] - g21fit(all_waves[gvals]), mask=~mask)
        ax[3].plot(all_waves[gvals], filtered_data2, rejsym, label="rejected data")
        ax[3].set_ylim(-0.5, 0.5)

    fname = f"fuv_mir_rv_fit_params_{args.wavereg}"
    if args.png:
        fig.savefig(f"{fname}.png")
    elif args.pdf:
        fig.savefig(f"{fname}.pdf")
    else:
        plt.show()
