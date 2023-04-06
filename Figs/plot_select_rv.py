import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

import numpy as np
import astropy.units as u

from dust_extinction.parameter_averages import CCM89, F19, D22, GCC09, G23

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

    fig, ax = plt.subplots(
        nrows=5,
        ncols=1,
        sharex=True,
        gridspec_kw={"height_ratios": [10, 2, 2, 2, 2]},
        figsize=(10, 10),
    )

    plt.subplots_adjust(wspace=0.0, hspace=0.0)

    modx = np.logspace(np.log10(0.0912), np.log10(32.0), 1000) * u.micron
    modx2 = np.logspace(np.log10(0.115), np.log10(3.0), 1000) * u.micron
    modx3 = np.logspace(np.log10(0.8), np.log10(4.0), 1000) * u.micron
    modx4 = np.logspace(np.log10(0.0912), np.log10(0.3), 1000) * u.micron
    rvs = [2.5, 3.1, 4.0, 5.5]
    colors = ["tomato", "olivedrab", "cornflowerblue", "blueviolet"]
    for rv, ccol in zip(rvs, colors):
        print(f"R(V) = {rv}")
        g22mod = G23(Rv=rv)
        ydata = g22mod(modx)
        ax[0].plot(modx, ydata, label=f"R(V) = {rv:.1f}", color=ccol)

        if rv == 5.5:
            ltxt = ["CCM89", "GCC09", "F19", "D22"]
        else:
            ltxt = [None, None, None, None]

        ccm89mod = CCM89(Rv=rv)
        # deviations
        dev = (ccm89mod(modx2) - g22mod(modx2)) / g22mod(modx2)
        print("CCM89 to G23", np.max(dev), np.average(dev))

        ydata = ccm89mod(modx2)
        ax[0].plot(
            modx2,
            ydata,
            linestyle="dashed",
            color=ccol,
            alpha=0.5,
            label=ltxt[0],
        )
        ax[1].plot(
            modx2,
            dev,
            # linestyle="dashed",
            color=ccol,
            alpha=0.8,
            label=f"ave = {np.average(dev):4.2f}, max = {np.max(np.absolute(dev)):4.2f}   (R(V) = {rv:3.1f})",
        )
        ax[1].text(0.1, 0.3, "CCM89", alpha=0.5)

        gcc09mod = GCC09(Rv=rv)
        # deviations
        dev = (gcc09mod(modx4) - g22mod(modx4)) / g22mod(modx4)
        print("GCC09 to G23", max(dev), np.average(dev))

        ydata = gcc09mod(modx4)
        ax[0].plot(
            modx4,
            ydata,
            linestyle="dashdot",
            color=ccol,
            alpha=0.5,
            label=ltxt[1],
        )
        ax[2].plot(
            modx4,
            dev,
            # linestyle="dashdot",
            color=ccol,
            alpha=0.8,
            label=f"ave = {np.average(dev):4.2f}, max = {np.max(np.absolute(dev)):4.2f}   (R(V) = {rv:3.1f})",
        )
        ax[2].text(0.1, 0.3, "GCC09", alpha=0.5)

        f19mod = F19(Rv=rv)
        # deviations
        dev = (f19mod(modx2) - g22mod(modx2)) / g22mod(modx2)
        print("F19 to G23", max(dev), np.average(dev))

        ydata = f19mod(modx2)
        ax[0].plot(
            modx2,
            ydata,
            linestyle="dotted",
            color=ccol,
            alpha=0.5,
            label=ltxt[2],
        )
        ax[3].plot(
            modx2,
            dev,
            # linestyle="dotted",
            color=ccol,
            alpha=0.8,
            label=f"ave = {np.average(dev):4.2f}, max = {np.max(np.absolute(dev)):4.2f}   (R(V) = {rv:3.1f})",
        )
        ax[3].text(0.1, 0.3, "F19", alpha=0.5)

        d22mod = D22(Rv=rv)
        # deviations
        dev = (d22mod(modx3) - g22mod(modx3)) / g22mod(modx3)
        print("D22 to G23", max(dev), np.average(dev))

        ydata = d22mod(modx3)
        ax[0].plot(
            modx3,
            ydata,
            linestyle=(0, (3, 1, 1, 1, 1, 1)),
            color=ccol,
            alpha=0.5,
            label=ltxt[3],
        )
        ax[4].plot(
            modx3,
            dev,
            # linestyle=(0, (3, 1, 1, 1, 1, 1)),
            color=ccol,
            alpha=0.8,
            label=f"ave = {np.average(dev):4.2f}, max = {np.max(np.absolute(dev)):4.2f}   (R(V) = {rv:3.1f})",
        )
        ax[4].text(0.1, 0.3, "D22", alpha=0.5)

    tax = ax[4]
    tax.set_xlabel(r"$\lambda$ [$\mu$m]")
    tax.set_xscale("log")
    tax.xaxis.set_major_formatter(ScalarFormatter())
    tax.xaxis.set_minor_formatter(ScalarFormatter())
    xticks = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]
    tax.set_xticks(xticks, minor=True)
    tax.tick_params(axis="x", which="minor", labelsize=fontsize * 0.8)
    tax.set_xlim(0.08, 35.0)

    ax[0].set_yscale("log")
    for k in range(1, 5):
        ax[k].plot([0.09, 4.0], [0.0, 0.0], "k--", alpha=0.5)
        ax[k].legend(fontsize=0.67 * fontsize, frameon=False, handlelength=2)
        ax[k].set_ylim(-0.75, 0.75)

    ax[0].yaxis.set_major_formatter(ScalarFormatter())

    # ax.set_ylabel(r"fractional deviation")
    ax[0].set_ylabel(r"$A(\lambda)/A(V)$")
    fig.text(0.02, 0.3, "Fractional Deviation", va="center", rotation="vertical")

    ax[0].legend(ncol=2, handlelength=4)

    ax[0].text(
        1.3,
        5.0,
        "Average",
        rotation="vertical",
        fontsize=0.7 * fontsize,
        verticalalignment="center",
        alpha=0.5,
    )
    ax[0].text(
        1.47,
        5.0,
        "Grain Size",
        rotation="vertical",
        fontsize=0.7 * fontsize,
        verticalalignment="center",
        alpha=0.5,
    )

    ax[0].arrow(
        1.7, 10.0, 0.0, -7.5, color="k", head_width=0.1, head_length=0.3, alpha=0.5
    )

    leg = ax[0].get_legend()
    for k in range(4, 8):
        leg.legendHandles[k].set_color("black")

    fig.tight_layout(w_pad=0.0, h_pad=0.0)

    fname = "fuv_mir_select_rv"
    if args.png:
        fig.savefig(f"{fname}.png")
    elif args.pdf:
        fig.savefig(f"{fname}.pdf")
    else:
        plt.show()
