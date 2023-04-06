import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u

from measure_extinction.extdata import ExtData
from dust_extinction.parameter_averages import CCM89, F19, GCC09, G23

from helpers import G22, G22MC, G22HF, G22LFnoweight


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rv", help="R(V) to plot", type=float, default=3.1)
    parser.add_argument("--drv", help="delta R(V) to plot", type=float, default=0.5)
    parser.add_argument(
        "--wavereg",
        help="wavelength region to plot",
        default="all",
        choices=["uv", "opt", "ir", "all"],
    )
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    # get the data
    files_gor09 = glob.glob("data/gor09*.fits")
    exts_gor09 = [ExtData(cfile) for cfile in files_gor09]
    color_gor09 = "blue"

    files_fit19 = glob.glob("data/fit19*.fits")
    exts_fit19 = [ExtData(cfile) for cfile in files_fit19]
    color_fit19 = "cyan"

    files_gor21 = glob.glob("data/gor21*.fits")
    exts_gor21 = [ExtData(cfile) for cfile in files_gor21]
    color_gor21 = "magenta"

    files_dec22 = glob.glob("data/dec22*.fits")
    exts_dec22 = [ExtData(cfile) for cfile in files_dec22]
    color_dec22 = "red"

    # setup plot
    fontsize = 14
    font = {"size": fontsize}
    plt.rc("font", **font)
    plt.rc("lines", linewidth=1.5)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("ytick.major", width=2)
    fig, ax = plt.subplots(figsize=(10, 8))

    all_exts = [exts_gor09, exts_fit19, exts_gor21, exts_dec22]
    all_colors = [color_gor09, color_fit19, color_gor21, color_dec22]

    sumrv = 0.0
    nrv = 0
    for exts, ccolor in zip(all_exts, all_colors):
        for iext in exts:
            rv = iext.columns["RV"]
            if np.absolute(rv[0] - args.rv) <= args.drv:
                sumrv += rv[0]
                nrv += 1
                iext.trans_elv_alav()
                iext.plot(ax, color=ccolor, alpha=0.2)
    print(sumrv / nrv)

    modx = np.logspace(np.log10(0.0912), np.log10(32.0), 1000) * u.micron
    modx2 = np.logspace(np.log10(0.115), np.log10(3.0), 1000) * u.micron
    rvs = [args.rv - args.drv, args.rv, args.rv + args.drv]
    for rv in rvs:
        g22mod = G23(Rv=rv)
        ax.plot(modx, g22mod(modx), color="black", lw=4, alpha=0.5, label="G23")

        # g22lfmod = G22LFnoweight(Rv=rv)
        # ax.plot(modx, g22lfmod(modx), color="magenta", alpha=0.5, linestyle="dotted", lw=2, label="G22LF")
        #
        # g22mcmod = G22MC(Rv=rv)
        # ax.plot(modx, g22mcmod(modx), color="green", alpha=0.5, linestyle="dotted", lw=2, label="G22MC")
        #
        # g22hfmod = G22HF(Rv=rv)
        # ax.plot(modx, g22hfmod(modx), color="blue", alpha=0.5, linestyle="dashed", lw=2, label="G22HF")

        ccm89mod = CCM89(Rv=rv)
        ax.plot(modx2, ccm89mod(modx2), linestyle="dashed", color="black", alpha=0.25, label="CCM89")

        f19mod = F19(Rv=rv)
        ax.plot(modx2, f19mod(modx2), linestyle="dotted", color="black", alpha=0.25, label="F19")

    ax.set_xscale("log")
    ax.set_yscale("log")

    if args.wavereg == "uv":
        ax.set_xlim(0.08, 0.35)
        ax.set_ylim(1.0, 15.0)
    elif args.wavereg == "ir":
        ax.set_xlim(0.8, 35.0)
        ax.set_ylim(0.001, 0.5)
    elif args.wavereg == "opt":
        ax.set_xlim(0.3, 1.0)
        ax.set_ylim(0.2, 3.0)

    ax.set_title(f"R(V) = {args.rv} +/- {args.drv}")

    ax.legend(ncol=2)

    fig.tight_layout()

    fname = f"fuv_mir_rep_curves_rv{args.rv}_drv{args.drv}_{args.wavereg}"
    if args.png:
        fig.savefig(f"{fname}.png")
    elif args.pdf:
        fig.savefig(f"{fname}.pdf")
    else:
        plt.show()
