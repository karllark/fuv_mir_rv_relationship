import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from astropy.table import QTable


def plot_irv_ssamp(ax, itab, label):
    gvals = itab["npts"] > 0
    for i in range(2):
        ax[0, i].plot(
            itab["waves"][gvals], itab["slopes"][gvals], label=label, alpha=0.75
        )
        ax[1, i].plot(
            itab["waves"][gvals], itab["intercepts"][gvals], label=label, alpha=0.75
        )
        ax[3, i].plot(
            itab["waves"][gvals], itab["sigmas"][gvals], label=label, alpha=0.75
        )


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
    fontsize = 12
    font = {"size": fontsize}
    plt.rc("font", **font)
    plt.rc("lines", linewidth=1.5)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("ytick.major", width=2)
    fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(16, 9), sharex="col", constrained_layout=True)

    # plot parameters
    plot_irv_ssamp(ax, gor09_fuse, "G09 FUSE")
    plot_irv_ssamp(ax, gor09_iue, "G09 IUE")
    # plot_irv_ssamp(ax, gor21_iue, "G21 IUE")
    # plot_irv_ssamp(ax, dec22_iue, "D22 IUE")
    plot_irv_ssamp(ax, fit19_stis, "F19")
    plot_irv_ssamp(ax, dec22_spexsxd, "D22 SpeXSXD")
    plot_irv_ssamp(ax, dec22_spexlxd, "D22 SpeXLXD")
    plot_irv_ssamp(ax, gor21_irs, "G21 IRS")

    ax[3, 0].set_xscale("log")
    ax[3, 1].set_xscale("log")

    ax[3, 0].set_xlim(0.09, 0.35)
    ax[3, 1].set_xlim(0.30, 20.0)

    ax[0, 0].set_ylim(0.0, 25.0)
    ax[1, 0].set_ylim(-2.0, 2.0)
    ax[3, 0].set_ylim(0.0, 2.0)
    ax[0, 1].set_ylim(-1.0, 3.0)
    ax[1, 1].set_ylim(-0.1, 1.1)
    ax[3, 1].set_ylim(0.0, 0.10)

    ax[3, 0].set_xlabel(r"$\lambda$ [$\mu$m]")
    ax[3, 1].set_xlabel(r"$\lambda$ [$\mu$m]")
    ax[0, 0].set_ylabel("slope")
    ax[1, 0].set_ylabel("intercept")
    ax[2, 0].set_ylabel("sigma")
    ax[3, 0].set_ylabel("scatter")

    ax[0, 1].axhline(linestyle="--", alpha=0.25, color="k", linewidth=2)
    ax[1, 1].axhline(linestyle="--", alpha=0.25, color="k", linewidth=2)
    ax[1, 0].axhline(linestyle="--", alpha=0.25, color="k", linewidth=2)

    ax[0, 1].legend(ncol=2)

    ax[3, 0].xaxis.set_major_formatter(ScalarFormatter())
    ax[3, 0].xaxis.set_minor_formatter(ScalarFormatter())
    ax[3, 0].set_xticks([0.09, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.35], minor=True)
    ax[3, 1].xaxis.set_major_formatter(ScalarFormatter())
    ax[3, 1].xaxis.set_minor_formatter(ScalarFormatter())
    ax[3, 1].set_xticks(
        [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0], minor=True
    )

    # plt.subplots_adjust(hspace=0)
    fig.tight_layout()

    fname = "fuv_mir_rv_fit_params"
    if args.png:
        fig.savefig(f"{fname}.png")
    elif args.pdf:
        fig.savefig(f"{fname}.pdf")
    else:
        plt.show()
