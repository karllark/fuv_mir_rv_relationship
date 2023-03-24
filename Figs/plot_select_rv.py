import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import astropy.units as u

from dust_extinction.parameter_averages import CCM89, F19, D22, GCC09

from helpers import G22

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", help="plot deviations", action="store_true")
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

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5.5))

    modx = np.logspace(np.log10(0.091), np.log10(34.0), 1000) * u.micron
    modx2 = np.logspace(np.log10(0.115), np.log10(3.0), 1000) * u.micron
    modx3 = np.logspace(np.log10(0.8), np.log10(4.0), 1000) * u.micron
    modx4 = np.logspace(np.log10(0.091), np.log10(0.3), 1000) * u.micron
    rvs = [2.5, 3.1, 4.0, 5.5]
    colors = ["tomato", "olivedrab", "cornflowerblue", "blueviolet"]
    for rv, ccol in zip(rvs, colors):
        print(f"R(V) = {rv}")
        g22mod = G22(Rv=rv)
        if args.dev:
            ydata = modx * 0.0
        else:
            ydata = g22mod(modx)
        ax.plot(modx, ydata, label=f"R(V) = {rv:.1f}", color=ccol)

        if rv == 5.5:
            ltxt = ["CCM89", "GCC09", "F19", "D22"]
        else:
            ltxt = [None, None, None, None]

        ccm89mod = CCM89(Rv=rv)
        # deviations
        dev = np.absolute(ccm89mod(modx2) - g22mod(modx2)) / g22mod(modx2)
        print("CCM89 to G23", max(dev), np.average(dev))

        if args.dev:
            ydata = dev
        else:
            ydata = ccm89mod(modx2)
        ax.plot(
            modx2,
            ydata,
            linestyle="dashed",
            color=ccol,
            alpha=0.5,
            label=ltxt[0],
        )

        gcc09mod = GCC09(Rv=rv)
        # deviations
        dev = np.absolute(gcc09mod(modx4) - g22mod(modx4)) / g22mod(modx4)
        print("GCC09 to G23", max(dev), np.average(dev))

        if args.dev:
            ydata = dev
        else:
            ydata = gcc09mod(modx4)
        ax.plot(
            modx4,
            ydata,
            linestyle="dashdot",
            color=ccol,
            alpha=0.5,
            label=ltxt[1],
        )

        f19mod = F19(Rv=rv)
        # deviations
        dev = np.absolute(f19mod(modx2) - g22mod(modx2)) / g22mod(modx2)
        print("F19 to G23", max(dev), np.average(dev))

        if args.dev:
            ydata = dev
        else:
            ydata = f19mod(modx2)
        ax.plot(
            modx2,
            ydata,
            linestyle="dotted",
            color=ccol,
            alpha=0.5,
            label=ltxt[2],
        )

        d22mod = D22(Rv=rv)
        # deviations
        dev = np.absolute(d22mod(modx3) - g22mod(modx3)) / g22mod(modx3)
        print("D22 to G23", max(dev), np.average(dev))

        if args.dev:
            ydata = dev
        else:
            ydata = d22mod(modx3)
        ax.plot(
            modx3,
            ydata,
            linestyle=(0, (3, 1, 1, 1, 1, 1)),
            color=ccol,
            alpha=0.5,
            label=ltxt[3],
        )

    ax.set_xscale("log")
    if not args.dev:
        ax.set_yscale("log")

    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(ScalarFormatter())
    xticks = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]
    ax.set_xticks(xticks, minor=True)
    ax.tick_params(axis="x", which="minor", labelsize=fontsize * 0.8)

    ax.yaxis.set_major_formatter(ScalarFormatter())

    ax.set_xlabel(r"$\lambda$ [$\mu$m]")
    if args.dev:
        ax.set_ylabel(r"fractional deviation")
    else:
        ax.set_ylabel(r"$A(\lambda)/A(V)$")

    ax.legend(ncol=2, handlelength=4)

    ax.text(
        1.3,
        5.0,
        "Average",
        rotation="vertical",
        fontsize=0.7 * fontsize,
        verticalalignment="center",
        alpha=0.5,
    )
    ax.text(
        1.47,
        5.0,
        "Grain Size",
        rotation="vertical",
        fontsize=0.7 * fontsize,
        verticalalignment="center",
        alpha=0.5,
    )

    if not args.dev:
        ax.arrow(1.7, 9.5, 0.0, -7.0, color="k", head_width=0.1, head_length=0.3, alpha=0.5)

    leg = ax.get_legend()
    for k in range(4, 8):
        leg.legendHandles[k].set_color("black")

    fig.tight_layout()

    fname = "fuv_mir_select_rv"
    if args.dev:
        fname = f"{fname}_dev"
    if args.png:
        fig.savefig(f"{fname}.png")
    elif args.pdf:
        fig.savefig(f"{fname}.pdf")
    else:
        plt.show()
