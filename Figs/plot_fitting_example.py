import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

from plot_rep_waves import cov_ellipse


def plot_2dcor_example(ax, x, y, xunc, yunc, corr, intercept, slope,
                       pcolor="tab:green"):
    """
    Show a plot giving an example data point with a model line for the
    2dcorr method including lines for standard y unc and ODR methods.
    """
    modx = np.arange(-5.1, 10.1, 0.1)
    mody = intercept + slope * modx

    cov = np.zeros((2, 2))
    cov[0, 0] = xunc ** 2
    cov[1, 1] = yunc ** 2
    cov[0, 1] = xunc * yunc * corr
    cov[1, 0] = cov[0, 1]
    ax.add_patch(cov_ellipse(x, y, cov, 1, color="k", alpha=0.25))

    # model line
    ax.plot(modx, mody, "k-", alpha=0.5)

    # fitting with y uncertainties
    ax.plot([x, x], [x, y], color="tab:blue", linestyle="dotted", alpha=0.5,
            label="y unc weighted")
    # ax.text(1.65, 2.75, r"8$\sigma$", color="tab:blue")

    # difference in x
    # ax.plot([x, 3.], [y, 3.], color="tab:purple", linestyle="dashed", alpha=0.5,
    #         label="ODR")
    # ax.text(3.5, 2.5, r"ODR", color="tab:purple")

    # 2d corr method
    datamod = multivariate_normal([x, y], cov)
    pos = np.column_stack((modx, mody))
    modvals = datamod.pdf(pos)
    maxmod = max(modvals)
    print(np.log10(maxmod))
    for k, cmod in enumerate(modvals):
        if cmod == maxmod:
            print(modx[k], mody[k], cmod)
            clabel = "2DCORR"
        else:
            clabel = None
        print(np.log10(cmod) - np.log10(maxmod))
        ax.plot([modx[k], x], [mody[k], y], color=pcolor,
                alpha=cmod/maxmod, label=clabel)

    ax.set_ylabel(r"x")
    ax.set_xlabel(r"y")

    ax.set_xlim(0.0, 5.0)
    ax.set_ylim(0.0, 5.0)

    # ax.legend()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        help="type of plot",
        default="data",
        choices=["data", "slope"],
    )
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

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    slope = 1.0
    intercept = 0.0

    # data point
    x = 2.
    y = 4.
    corr = 0.

    if args.type == "data":
        plot_2dcor_example(ax[0, 0], x, y, 1.0, 0.25, 0.7, intercept, slope)
        plot_2dcor_example(ax[1, 0], x, y, 1.0, 0.25, 0.7, intercept, slope)
        plot_2dcor_example(ax[1, 0], x, y, 1.0, 0.25, 0.7, 4.0, -1.0, pcolor="tab:blue")
        plot_2dcor_example(ax[1, 1], x, y, 1.0, 0.05, corr, intercept, slope)
        plot_2dcor_example(ax[0, 1], x, y, 0.05, 1.0, corr, intercept, slope)
    else:
        corr = 0.7
        plot_2dcor_example(ax[0, 0], x, y, 1.0, 0.25, corr, intercept, slope)
        plot_2dcor_example(ax[0, 1], x, y, 1.0, 0.25, corr, 3.0, slope)
        plot_2dcor_example(ax[1, 0], x, y, 1.0, 0.25, corr, 4.5, -0.5)
        plot_2dcor_example(ax[1, 1], x, y, 0.5, 0.5, 0.0, intercept, slope)

    fig.tight_layout()

    fname = f"fitting_2dcor_example_{args.type}"
    if args.png:
        fig.savefig(f"{fname}.png")
    elif args.pdf:
        fig.savefig(f"{fname}.pdf")
    else:
        plt.show()
