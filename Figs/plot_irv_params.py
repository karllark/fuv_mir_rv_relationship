import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

from astropy.table import QTable
import astropy.units as u
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling import Fittable1DModel, Parameter

# from dust_extinction.shapes import G21
from dust_extinction.helpers import _get_x_in_wavenumbers, _test_valid_x_range
from dust_extinction.shapes import _modified_drude


class G21mod(Fittable1DModel):
    r"""
    Gordon et al. (2021) powerlaw plus two modified Drude profiles
    (for the 10 & 20 micron silicate features)
    for the 1 to 40 micron A(lambda)/A(V) extinction curve.

    Parameters
    ----------
    scale: float
        amplitude of the powerlaw at 1 micron
    alpha: float
        power of powerlaw
    sil1_amp: float
        central amplitude of the 10 micron silicate feature
    sil1_amp: float
        central amplitude of the 10 micron silicate feature
    sil1_amp: float
        central amplitude of the 10 micron silicate feature
    sil1_amp: float
        central amplitude of the 10 micron silicate feature
    sil1_amp: float
        central amplitude of the 10 micron silicate feature

    Notes
    -----
    From Gordon et al. (2021, ApJ, submitted)

    Only applicable at NIR/MIR wavelengths from 1-40 micron

    Example showing a G21 curve with components identified.

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.shapes import G21

        fig, ax = plt.subplots()

        # generate the curves and plot them
        lam = np.logspace(np.log10(1.01), np.log10(39.9), num=1000)
        x = (1.0/lam)/u.micron

        ext_model = G21()
        ax.plot(1/x,ext_model(x),label='total')

        ext_model = G21(sil1_amp=0.0, sil2_amp=0.0)
        ax.plot(1./x,ext_model(x),label='power-law only')

        ext_model = G21(sil2_amp=0.0)
        ax.plot(1./x,ext_model(x),label='power-law+sil1 only')

        ext_model = G21(sil1_amp=0.0)
        ax.plot(1./x,ext_model(x),label='power-law+sil2 only')

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlabel('$\lambda$ [$\mu$m]')
        ax.set_ylabel('$A(x)/A(V)$')

        ax.set_title('G21')

        ax.legend(loc='best')
        plt.show()

    """

    # inputs = ("x",)
    # outputs = ("axav",)

    scale = Parameter(
        description="powerlaw: amplitude", default=0.37, bounds=(0.0, 1.0)
    )
    alpha = Parameter(description="powerlaw: alpha", default=1.7, bounds=(0.5, 5.0))
    alpha2 = Parameter(description="powerlaw: alpha2", default=1.4, bounds=(-0.5, 5.0))
    swave = Parameter(description="powerlaw: swave", default=4.0, bounds=(2.0, 10.0))
    ice_amp = Parameter(
        description="ice 3um: amplitude", default=0.0019, bounds=(0.0001, 0.3)
    )
    ice_center = Parameter(
        description="ice 3um: center", default=3.02, bounds=(2.9, 3.1)
    )
    ice_fwhm = Parameter(description="ice 3um: fwhm", default=0.45, bounds=(0.3, 0.6))
    ice_asym = Parameter(
        description="ice 3um: asymmetry", default=-1.0, bounds=(-2.0, 0.0)
    )
    sil1_amp = Parameter(
        description="silicate 10um: amplitude", default=0.07, bounds=(0.001, 0.3)
    )
    sil1_center = Parameter(
        description="silicate 10um: center", default=9.87, bounds=(8.0, 12.0)
    )
    sil1_fwhm = Parameter(
        description="silicate 10um: fwhm", default=2.5, bounds=(1.0, 10.0)
    )
    sil1_asym = Parameter(
        description="silicate 10um: asymmetry", default=-0.23, bounds=(-2.0, 2.0)
    )
    sil2_amp = Parameter(
        description="silicate 20um: amplitude", default=0.025, bounds=(0.001, 0.3)
    )
    sil2_center = Parameter(
        description="silicate 20um: center", default=17.0, bounds=(16.0, 24.0)
    )
    sil2_fwhm = Parameter(
        description="silicate 20um: fwhm", default=13.0, bounds=(5.0, 20.0)
    )
    sil2_asym = Parameter(
        description="silicate 20um: asymmetry", default=-0.27, bounds=(-2.0, 2.0)
    )

    x_range = [1.0 / 40.0, 1.0 / 0.8]

    def evaluate(
        self,
        in_x,
        scale,
        alpha,
        alpha2,
        swave,
        ice_amp,
        ice_center,
        ice_fwhm,
        ice_asym,
        sil1_amp,
        sil1_center,
        sil1_fwhm,
        sil1_asym,
        sil2_amp,
        sil2_center,
        sil2_fwhm,
        sil2_asym,
    ):
        """
        G21 function

        Parameters
        ----------
        in_x: float
           expects either x in units of wavelengths or frequency
           or assumes wavelengths in wavenumbers [1/micron]

        Returns
        -------
        axav: np array (float)
            A(x)/A(V) extinction curve [mag]

        Raises
        ------
        ValueError
           Input x values outside of defined range
        """
        x = _get_x_in_wavenumbers(in_x)

        # check that the wavenumbers are within the defined range
        _test_valid_x_range(x, self.x_range, "G21")

        wave = 1 / x

        # broken powerlaw
        # swave = 4.0
        axav = scale * (wave ** (-1.0 * alpha))
        (gindxs,) = np.where(wave > swave)
        if len(gindxs) > 0:
            norm_ratio = swave ** (-1.0 * alpha) / swave ** (-1.0 * alpha2)
            axav[gindxs] = scale * norm_ratio * (wave[gindxs] ** (-1.0 * alpha2))

        # silicate feature drudes
        axav += _modified_drude(wave, ice_amp, ice_center, ice_fwhm, ice_asym)
        axav += _modified_drude(wave, sil1_amp, sil1_center, sil1_fwhm, sil1_asym)
        axav += _modified_drude(wave, sil2_amp, sil2_center, sil2_fwhm, sil2_asym)

        return axav


def plot_irv_ssamp(ax, itab, label, color="k", simpfit=False, inst=None, ncol=2):

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
    for i in range(ncol):
        if simpfit:
            ax[1, i].plot(
                itab["waves"][gvals],
                itab["slopes"][gvals],
                linestyle="dashed",
                color=color,
                alpha=0.75,
            )
            ax[0, i].plot(
                itab["waves"][gvals],
                itab["intercepts"][gvals],
                linestyle="dashed",
                color=color,
                alpha=0.75,
            )
            if "rmss" in itab.colnames:
                ax[3, i].plot(
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
            ax[1, i].plot(
                itab["waves"][gvals],
                itab["hfslopes"][gvals],
                color=color,
                label=label,
                alpha=0.75,
            )
            ax[0, i].plot(
                itab["waves"][gvals],
                itab["hfintercepts"][gvals],
                color=color,
                label=label,
                alpha=0.75,
            )
            ax[2, i].plot(
                itab["waves"][gvals],
                itab["hfsigmas"][gvals],
                color=color,
                label=label,
                alpha=0.75,
            )
            # ax[3, i].plot(
            #     itab["waves"][gvals],
            #     itab["hfrmss"][gvals],
            #     color=color,
            #     label=label,
            #     alpha=0.75,
            # )

    return (itab["npts"], itab["waves"], itab["hfintercepts"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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

    # setup plot
    fontsize = 14
    font = {"size": fontsize}
    plt.rc("font", **font)
    plt.rc("lines", linewidth=1.5)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("ytick.major", width=2)
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(16, 12), sharex="col")

    # plot parameters
    plot_irv_ssamp(ax, gor09_fuse, "G09 FUSE", color="blue")
    plot_irv_ssamp(ax, gor09_iue, "G09 IUE", color="orange", inst="IUE")
    plot_irv_ssamp(ax, gor21_iue, "G21 IUE", color="yellow", inst="IUE")
    plot_irv_ssamp(ax, dec22_iue, "D22 IUE", color="purple", inst="IUE")
    plot_irv_ssamp(ax, fit19_stis, "F19", color="green", inst="STIS")
    dec22_res1 = plot_irv_ssamp(ax, dec22_spexsxd, "D22 SpeXSXD", color="red")
    dec22_res2 = plot_irv_ssamp(
        ax, dec22_spexlxd, "D22 SpeXLXD", color="purple", inst="SpeXLXD"
    )
    gor21_res = plot_irv_ssamp(ax, gor21_irs, "G21 IRS", color="brown")

    ax[2, 0].set_xscale("log")
    ax[2, 1].set_xscale("log")

    ax[2, 0].set_xlim(0.09, 0.35)
    ax[2, 1].set_xlim(0.35, 35.0)

    ax[1, 0].set_ylim(0.0, 50.0)
    ax[0, 0].set_ylim(1.0, 7.5)
    ax[2, 0].set_ylim(0.0, 1.5)
    # ax[3, 0].set_ylim(0.0, 2.0)
    ax[1, 1].set_ylim(-2.0, 5.0)
    ax[0, 1].set_yscale("log")
    ax[0, 1].set_ylim(0.01, 2.0)
    ax[2, 1].set_ylim(0.0, 0.1)
    # ax[3, 1].set_ylim(0.0, 0.10)

    ax[2, 0].set_xlabel(r"$\lambda$ [$\mu$m]")
    ax[2, 1].set_xlabel(r"$\lambda$ [$\mu$m]")
    ax[1, 0].set_ylabel("slope")
    ax[0, 0].set_ylabel("intercept")
    ax[2, 0].set_ylabel("sigma")
    # ax[3, 0].set_ylabel("scatter")

    ax[0, 1].axhline(linestyle="--", alpha=0.25, color="k", linewidth=2)
    ax[1, 1].axhline(linestyle="--", alpha=0.25, color="k", linewidth=2)
    ax[1, 0].axhline(linestyle="--", alpha=0.25, color="k", linewidth=2)

    ax[0, 0].legend(ncol=2)

    ax[2, 0].xaxis.set_major_formatter(ScalarFormatter())
    ax[2, 0].xaxis.set_minor_formatter(ScalarFormatter())
    ax[2, 0].set_xticks([0.09, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.35], minor=True)
    ax[2, 1].xaxis.set_major_formatter(ScalarFormatter())
    ax[2, 1].xaxis.set_minor_formatter(ScalarFormatter())
    ax[2, 1].set_xticks(
        [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0], minor=True
    )

    # fit G21 to > 1 micron data
    all_npts = np.concatenate((dec22_res1[0], dec22_res2[0], gor21_res[0]))
    all_waves = np.concatenate((dec22_res1[1], dec22_res2[1], gor21_res[1]))
    all_intercepts = np.concatenate((dec22_res1[2], dec22_res2[2], gor21_res[2]))
    g21init = G21mod()
    g21init.ice_amp.fixed = True
    g21init.ice_fwhm.fixed = True
    g21init.ice_center.fixed = True
    g21init.ice_asym.fixed = True
    # g21init.sil1_asym.fixed = True
    fit = LevMarLSQFitter()
    bvals = all_waves < 0.8 * u.micron
    all_npts[bvals] = 0
    gvals = all_npts > 0
    g21fit = fit(
        g21init, 1.0 / all_waves[gvals].value, all_intercepts[gvals], maxiter=500
    )
    print(g21fit.param_names)
    print(g21fit.parameters)
    print(g21init.parameters)
    print(fit.fit_info["message"])
    ax[0, 1].plot(all_waves[gvals], g21fit(all_waves[gvals]))
    # lam = np.logspace(np.log10(1.01), np.log10(39.9), num=1000)
    # x = (1.0 / lam) / u.micron
    # ax[0, 1].plot(x, g21init(x))

    # plt.subplots_adjust(hspace=0)
    fig.tight_layout()

    fname = "fuv_mir_rv_fit_params"
    if args.png:
        fig.savefig(f"{fname}.png")
    elif args.pdf:
        fig.savefig(f"{fname}.pdf")
    else:
        plt.show()
