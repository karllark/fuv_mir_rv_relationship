import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from matplotlib.colors import LinearSegmentedColormap

from plot_rep_waves import cov_ellipse


def CustomCmap(from_rgb, to_rgb):
    # from color r,g,b
    r1, g1, b1 = from_rgb
    # to color r,g,b
    r2, g2, b2 = to_rgb

    cdict = {
        "red": ((0, r1, r1), (1, r2, r2)),
        "green": ((0, g1, g1), (1, g2, g2)),
        "blue": ((0, b1, b1), (1, b2, b2)),
    }

    cmap = LinearSegmentedColormap("custom_cmap", cdict)
    return cmap


def plot_2dcor_example(
    ax, x, y, xunc, yunc, corr, intercept, slope, pcolor="tab:green"
):
    """
    Show a plot giving an example data point with a model line for the
    2dcorr method including lines for standard y unc and ODR methods.
    """
    modx = np.arange(-5.1, 10.1, 0.1)
    mody = intercept + slope * modx

    cov = np.zeros((2, 2))
    cov[0, 0] = xunc**2
    cov[1, 1] = yunc**2
    cov[0, 1] = xunc * yunc * corr
    cov[1, 0] = cov[0, 1]
    ax.add_patch(cov_ellipse(x, y, cov, 1, color="k", alpha=0.25))

    # model line
    ax.plot(modx, mody, "k-", alpha=0.5)

    # 2d corr method
    datamod = multivariate_normal([x, y], cov)
    pos = np.column_stack((modx, mody))
    modvals = datamod.pdf(pos)
    maxmod = max(modvals)
    minmod = min(modvals[modvals > 0.0])
    # print(minmod, maxmod)
    # minmod = 1e-10
    # print(np.log10(maxmod))
    for k, cmod in enumerate(modvals):
        if cmod == maxmod:
            # print(modx[k], mody[k], cmod)
            clabel = "2DCORR"
        else:
            clabel = None
        if np.isfinite(cmod) & (cmod > minmod):
            # calpha = (np.log10(cmod) - np.log10(minmod)) / (np.log10(maxmod) - np.log10(minmod))
            calpha = cmod / maxmod
            ax.plot(
                [modx[k], x], [mody[k], y], color=pcolor, alpha=calpha, label=clabel
            )

    # difference in x
    ax.plot(
        [x, 3.0],
        [y, 3.0],
        color="tab:purple",
        linestyle="dashed",
        alpha=0.5,
        label="ODR",
    )
    # ax.text(3.5, 2.5, r"ODR", color="tab:purple")

    # fitting with y uncertainties
    ax.plot(
        [x, x],
        [x, y],
        color="tab:blue",
        linestyle="dotted",
        alpha=0.5,
        label="y unc only",
    )
    # ax.text(1.65, 2.75, r"8$\sigma$", color="tab:blue")

    ax.set_xlabel(r"x")
    ax.set_ylabel(r"y")

    ax.set_xlim(0.0, 6.0)
    ax.set_ylim(0.0, 6.0)

    # plt.colorbar(cmap, cax=ax)


#    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
#                                    norm=norm,
#                                    orientation='horizontal')
#    cb1.set_label('Some Units')

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

    fig, ax = plt.subplots(
        nrows=2, ncols=2, figsize=(6, 10), gridspec_kw={"width_ratios": [15, 1]}
    )

    slope = 1.0
    intercept = 0.0

    # data point
    x = 2.0
    y = 4.0
    corr = 0.0

    if args.type == "data":
        plot_2dcor_example(ax[0, 0], x, y, 1.0, 0.25, 0.7, intercept, slope)
        plot_2dcor_example(ax[1, 0], x, y, 0.05, 1.0, corr, intercept, slope)
        # plot_2dcor_example(ax[0, 0], x, y, 1.0, 0.25, 0.7, intercept, slope)
        # plot_2dcor_example(ax[1, 0], x, y, 1.0, 0.25, 0.7, intercept, slope)
        # plot_2dcor_example(ax[1, 0], x, y, 1.0, 0.25, 0.7, 4.0, -1.0, pcolor="tab:blue")
        # plot_2dcor_example(ax[1, 1], x, y, 1.0, 0.05, corr, intercept, slope)
        # plot_2dcor_example(ax[0, 1], x, y, 0.05, 1.0, corr, intercept, slope)
    else:
        corr = 0.7
        plot_2dcor_example(ax[0, 0], x, y, 1.0, 0.25, corr, intercept, slope)
        plot_2dcor_example(ax[0, 1], x, y, 1.0, 0.25, corr, 3.0, slope)
        plot_2dcor_example(ax[1, 0], x, y, 1.0, 0.25, corr, 4.5, -0.5)
        plot_2dcor_example(ax[1, 1], x, y, 0.5, 0.5, 0.0, intercept, slope)

    ax[0, 0].legend()

    props = dict(facecolor="black", alpha=0.05)
    ax[0, 0].text(0.3, 5.5, "x & y uncs", bbox=props)
    ax[1, 0].text(0.3, 5.5, "y >> x uncs", bbox=props)

    cmap = CustomCmap([1.00, 1.00, 1.00], [44.0 / 256.0, 160.0 / 256.0, 44.0 / 256.0])

    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

    cb = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax[0, 1],
    )
    cb.set_label(label="Normalized Probability", size=0.8 * fontsize)
    cb.ax.tick_params(labelsize=0.8 * fontsize)

    cb = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax[1, 1],
    )
    cb.set_label(label="Normalized Probability", size=0.8 * fontsize)
    cb.ax.tick_params(labelsize=0.8 * fontsize)

    fig.tight_layout()

    fname = f"fitting_2dcor_example_{args.type}"
    if args.png:
        fig.savefig(f"{fname}.png")
    elif args.pdf:
        fig.savefig(f"{fname}.pdf")
    else:
        plt.show()
