import argparse
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from astropy.stats import sigma_clip
from astropy.modeling import models, fitting
from astropy.modeling import Fittable1DModel, Parameter
from astropy.modeling.models import Polynomial1D


class BrokenLine(Fittable1DModel):
    intercept = Parameter()
    slope1 = Parameter()
    slope2 = Parameter()
    breakval = Parameter()

    @staticmethod
    def evaluate(x, intercept, slope1, slope2, breakval):

        y = intercept + slope1 * x
        gvals = x > breakval
        line2_diff = (intercept + slope1 * breakval) - (intercept + slope2 * breakval)
        y[gvals] = line2_diff + intercept + slope2 * x[gvals]

        return y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="name of file")
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    fontsize = 12

    font = {"size": fontsize}

    plt.rc("font", **font)

    plt.rc("lines", linewidth=1)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("ytick.major", width=2)

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 8), sharex=True)

    # read in the data
    a = Table.read(args.file, format="ascii.commented_header")

    sindxs = np.argsort(a["irv"])
    x = a["irv"][sindxs]
    xunc = a["irv_unc"][sindxs]
    y = a["alav"][sindxs]
    yunc = a["alav_unc"][sindxs]

    # do the simple linear fit
    tax = ax[0, 0]
    tax.errorbar(x, y, yerr=yunc, fmt="ko", label="Fitted Data")
    fit = fitting.LinearLSQFitter()
    line_init = models.Linear1D()
    fitted_line = fit(line_init, x, y, weights=1.0 / yunc)
    tax.plot(x, fitted_line(x), "k-", label="Linear Fit", lw=4.0, alpha=0.5)
    tax.set_ylabel(r"A($\lambda$)/A(V)")
    tax.legend()

    tax = ax[1, 0]
    tax.errorbar(x, y - fitted_line(x), yerr=yunc, fmt="ko", label="Fitted Data")
    tax.axhline(0.0, linestyle="dashed", alpha=0.5, color="black", lw=4.0)
    tax.set_ylabel(r"A($\lambda$)/A(V) - Fit")
    tax.set_xlabel("1/R(V) - 1/3.1")
    tax.legend()

    # fitting with outlier removal
    tax = ax[0, 1]
    or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=3, sigma=3.0)
    fitted_line, mask = or_fit(line_init, x, y, weights=1.0 / yunc)
    filtered_data = np.ma.masked_array(y, mask=mask)

    print(fitted_line)

    tax.errorbar(
        x, y, xerr=xunc, yerr=yunc, fmt="ko", fillstyle="none", label="Clipped Data"
    )
    tax.plot(x, filtered_data, "ko", label="Fitted Data")
    tax.plot(x, fitted_line(x), "k-", label="Linear Fit", lw=4.0, alpha=0.5)
    tax.legend()

    tax = ax[1, 1]
    tax.errorbar(
        x,
        y - fitted_line(x),
        yerr=yunc,
        fmt="ko",
        fillstyle="none",
        label="Clipped Data",
    )
    tax.plot(x, filtered_data - fitted_line(x), "ko", label="Fitted Data")
    tax.axhline(0.0, linestyle="dashed", alpha=0.5, color="black", lw=4.0)
    tax.set_xlabel("1/R(V) - 1/3.1")
    tax.legend()

    # fitting brokenline with outlier removal
    tax = ax[0, 2]
    # line_init = BrokenLine(
    #     intercept=fitted_line.intercept,
    #     slope1=fitted_line.slope * 0.5,
    #     slope2=fitted_line.slope * 2.0,
    #     breakval=-0.08,
    # )
    # line_init.breakval.fixed = True

    line_init = Polynomial1D(2)
    # line_init.c1 = 0.0
    # line_init.c1.fixed = True

    fitx = x

    # fit = fitting.LevMarLSQFitter()
    fit = fitting.LinearLSQFitter()
    or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=3, sigma=3.0)
    fitted_line, mask = or_fit(line_init, fitx, y, weights=1.0 / yunc)
    filtered_data = np.ma.masked_array(y, mask=mask)

    print(fitted_line)

    tax.errorbar(x, y, yerr=yunc, fmt="ko", fillstyle="none", label="Clipped Data")
    tax.plot(x, filtered_data, "ko", label="Fitted Data")
    tax.plot(x, fitted_line(fitx), "k-", label="a + bx + cx^2 Fit", lw=4.0, alpha=0.5)
    tax.legend()

    tax = ax[1, 2]
    tax.errorbar(
        x,
        y - fitted_line(fitx),
        yerr=yunc,
        fmt="ko",
        fillstyle="none",
        label="Clipped Data",
    )
    tax.plot(x, filtered_data - fitted_line(fitx), "ko", label="Fitted Data")
    tax.axhline(0.0, linestyle="dashed", alpha=0.5, color="black", lw=4.0)
    tax.set_xlabel("1/R(V) - 1/3.1")
    tax.legend()

    fig.suptitle(args.file)

    fig.tight_layout()

    fname = args.file.replace(".dat", "")
    if args.png:
        fig.savefig(f"{fname}.png")
    elif args.pdf:
        fig.savefig(f"{fname}.pdf")
    else:
        plt.show()
