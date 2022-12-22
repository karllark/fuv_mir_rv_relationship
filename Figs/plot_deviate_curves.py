import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import astropy.units as u
from astropy.table import Table

from measure_extinction.extdata import ExtData
from helpers import G22

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    fontsize = 14

    font = {"size": fontsize}

    plt.rc("font", **font)

    plt.rc("lines", linewidth=2)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("ytick.major", width=2)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 7), sharex=True, sharey=True)

    modx = np.logspace(np.log10(0.115), np.log10(3.0), 1000) * u.micron

    # show MW example high UV
    ax = axs[0, 1]
    cfile = "data/fit19_HD62542_ext.fits"
    a = ExtData(cfile)
    a.trans_elv_alav()
    a.plot(ax, color="blue", legend_key="STIS", legend_label="MW HD62542")
    rv = a.columns["RV"][0]
    g22mod = G22(Rv=rv)
    ax.plot(modx, g22mod(modx), label=f"R(V) = {rv:.2f}", color="black", alpha=0.5)
    ax.legend(handlelength=2)
    ax.set_xlabel("")

    # show MW example low UV
    ax = axs[0, 0]
    cfile = "data/dense/gor21_hd283809_hd064802_ext.fits"
    a = ExtData(cfile)
    a.trans_elv_alav()
    a.plot(ax, color="blue", legend_key="IUE", legend_label="MW HD283809")
    # cfile = "data/dense/dec22_hd283809_hd003360_ext.fits"
    # a = ExtData(cfile)
    # a.trans_elv_alav()
    # a.plot(ax, color="red", legend_key="IUE", legend_label="MW HD283809")

    rv = a.columns["RV"][0]
    g22mod = G22(Rv=rv)
    ax.plot(modx, g22mod(modx), label=f"R(V) = {rv:.2f}", color="black", alpha=0.5)
    ax.legend(ncol=1, handlelength=2)
    ax.set_xlabel("")

    # show LMC Example
    ax = axs[1, 0]

    a = Table.read(
        "data/MCs/lmc2_ext.dat",
        format="ascii.basic",
        header_start=4,
        names=["wave", "alav", "unc"],
    )
    ax.plot(
        1.0 / a["wave"][8:-3],
        a["alav"][8:-3],
        color="blue",
        linestyle="solid",
        label="LMC 2 Supershell Average",
    )
    ax.plot(1.0 / a["wave"][0:8], a["alav"][0:8], "bo", mfc="white")

    rv = 2.76
    g22mod = G22(Rv=rv)
    ax.plot(modx, g22mod(modx), label=f"R(V) = {rv:.2f}", color="black", alpha=0.5)
    ax.legend(ncol=1, handlelength=2)

    # show SMC Example(s)
    ax = axs[1, 1]

    a = Table.read(
        "data/MCs/smcbar_ext.dat",
        format="ascii.basic",
        header_start=4,
        names=["wave", "alav", "unc"],
    )
    ax.plot(
        1.0 / a["wave"][8:-3],
        a["alav"][8:-3],
        color="blue",
        linestyle="solid",
        label="SMC Bar Aveage",
    )
    ax.plot(1.0 / a["wave"][0:8], a["alav"][0:8], "bo", mfc="white")

    rv = 2.74
    g22mod = G22(Rv=rv)
    ax.plot(modx, g22mod(modx), label=f"R(V) = {rv:.2f}", color="black", alpha=0.5)
    ax.legend(handlelength=2)

    # apply to all
    axs[0, 0].set_xscale("log")
    axs[0, 0].xaxis.set_major_formatter(ScalarFormatter())
    axs[0, 0].set_xlim(0.09, 3.0)

    # ax.xaxis.set_minor_formatter(ScalarFormatter())
    # xticks = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]
    # ax.set_xticks(xticks, minor=True)
    # ax.tick_params(axis="x", which="minor", labelsize=fontsize * 0.8)

    axs[0, 0].yaxis.set_major_formatter(ScalarFormatter())

    axs[1, 0].set_xlabel(r"$\lambda$ [$\mu$m]")
    axs[1, 1].set_xlabel(r"$\lambda$ [$\mu$m]")
    axs[0, 0].set_ylabel(r"$A(\lambda)/A(V)$")
    axs[1, 0].set_ylabel(r"$A(\lambda)/A(V)$")

    fig.tight_layout()

    fname = "fuv_mir_deviate_curves"
    if args.png:
        fig.savefig(f"{fname}.png")
    elif args.pdf:
        fig.savefig(f"{fname}.pdf")
    else:
        plt.show()
