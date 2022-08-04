import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import astropy.units as u

from dust_extinction.parameter_averages import CCM89, F19, D22, GCC09

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

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5.5))

    modx = np.logspace(np.log10(0.091), np.log10(34.0), 1000) * u.micron
    modx2 = np.logspace(np.log10(0.115), np.log10(3.0), 1000) * u.micron
    modx3 = np.logspace(np.log10(0.8), np.log10(4.0), 1000) * u.micron
    modx4 = np.logspace(np.log10(0.091), np.log10(0.3), 1000) * u.micron
    rvs = [2.5, 3.1, 4.0, 5.5]
    colors = ["tomato", "olivedrab", "cornflowerblue", "blueviolet"]
    for rv, ccol in zip(rvs, colors):
        g22mod = G22(Rv=rv)
        ax.plot(modx, g22mod(modx), label=f"R(V) = {rv:.1f}", color=ccol)

        if rv == 5.5:
            ltxt = ["CCM89", "GCC09", "F19", "D22"]
        else:
            ltxt = [None, None, None, None]

        ccm89mod = CCM89(Rv=rv)
        ax.plot(
            modx2,
            ccm89mod(modx2),
            linestyle="dashed",
            color=ccol,
            alpha=0.5,
            label=ltxt[0],
        )

        gcc09mod = GCC09(Rv=rv)
        ax.plot(
            modx4,
            gcc09mod(modx4),
            linestyle="dashdot",
            color=ccol,
            alpha=0.5,
            label=ltxt[1],
        )

        f19mod = F19(Rv=rv)
        ax.plot(
            modx2,
            f19mod(modx2),
            linestyle="dotted",
            color=ccol,
            alpha=0.5,
            label=ltxt[2],
        )

        d22mod = D22(Rv=rv)
        ax.plot(
            modx3,
            d22mod(modx3),
            linestyle=(0, (5, 7)),
            color=ccol,
            alpha=0.5,
            label=ltxt[3],
        )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(ScalarFormatter())
    xticks = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]
    ax.set_xticks(xticks, minor=True)
    ax.tick_params(axis="x", which="minor", labelsize=fontsize * 0.8)

    ax.yaxis.set_major_formatter(ScalarFormatter())

    ax.set_xlabel(r"$\lambda$ [$\mu$m]")
    ax.set_ylabel(r"$A(\lambda)/A(V)$")

    ax.legend(ncol=2, handlelength=4)

    leg = ax.get_legend()
    for k in range(4, 8):
        leg.legendHandles[k].set_color("black")

    fig.tight_layout()

    fname = "fuv_mir_select_rv"
    if args.png:
        fig.savefig(f"{fname}.png")
    elif args.pdf:
        fig.savefig(f"{fname}.pdf")
    else:
        plt.show()
