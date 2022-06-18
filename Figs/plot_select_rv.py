import argparse
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u

from dust_extinction.parameter_averages import CCM89, F19

from helpers import G22

if __name__ == '__main__':
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

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5.5))

    modx = np.logspace(np.log10(0.091), np.log10(34.0), 1000) * u.micron
    modx2 = np.logspace(np.log10(0.115), np.log10(3.0), 1000) * u.micron
    rvs = [2.0, 2.5, 3.1, 4.0, 5.5, 6.0]
    for rv in rvs:
        g22mod = G22(Rv=rv)
        ax.plot(modx, g22mod(modx), label=f"R(V) = {rv:.1f}")

        ccm89mod = CCM89(Rv=rv)
        ax.plot(modx2, ccm89mod(modx2), linestyle="dashed")

        f19mod = F19(Rv=rv)
        ax.plot(modx2, f19mod(modx2), linestyle="dotted")

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.legend()

    fig.tight_layout()

    fname = "fuv_mir_select_rv"
    if args.png:
        fig.savefig(f"{fname}.png")
    elif args.pdf:
        fig.savefig(f"{fname}.pdf")
    else:
        plt.show()
