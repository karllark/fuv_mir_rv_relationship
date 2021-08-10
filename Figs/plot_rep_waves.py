import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np

import astropy.units as u

from measure_extinction.extdata import ExtData


def get_alav(exts, src, wave):

    n_exts = len(exts)
    oext = np.full((n_exts, 2), np.nan)
    for i, iext in enumerate(exts):
        if src in iext.waves.keys():
            sindxs = np.argsort(np.absolute(iext.waves[src] - wave))
            if (iext.npts[src][sindxs[0]] > 0) and (iext.exts[src][sindxs[0]] > 0):
                oext[i, 0] = iext.exts[src][sindxs[0]]
                oext[i, 1] = iext.uncs[src][sindxs[0]]
            else:
                oext[i, 0] = np.nan
                oext[i, 1] = np.nan

    return oext


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rv", help="plot versus R(V)", action="store_true")
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    # read in all the extinction curves
    files_val04 = glob.glob("data/val04*.fits")
    exts_val04 = [ExtData(cfile) for cfile in files_val04]
    psym_val04 = "go"

    files_gor09 = glob.glob("data/gor09*.fits")
    exts_gor09 = [ExtData(cfile) for cfile in files_gor09]
    psym_gor09 = "bs"

    files_fit19 = glob.glob("data/fit19*.fits")
    exts_fit19 = [ExtData(cfile) for cfile in files_fit19]
    psym_fit19 = "cv"

    files_gor21 = glob.glob("data/gor21*.fits")
    exts_gor21 = [ExtData(cfile) for cfile in files_gor21]
    psym_gor21 = "m^"

    files_dec22 = glob.glob("data/decleir22/*.fits")
    exts_dec22 = [ExtData(cfile) for cfile in files_dec22]
    psym_dec22 = "r>"

    # get R(V) values
    n_gor09 = len(files_gor09)
    rvs_gor09 = np.zeros((n_gor09, 2))
    for i, iext in enumerate(exts_gor09):
        irv = iext.columns["RV"]
        rvs_gor09[i, 0] = irv[0]
        rvs_gor09[i, 1] = irv[1]
        iext.trans_elv_alav()

    n_val04 = len(files_val04)
    rvs_val04 = np.zeros((n_val04, 2))
    for i, iext in enumerate(exts_val04):
        irv = iext.columns["RV"]
        rvs_val04[i, 0] = irv[0]
        rvs_val04[i, 1] = irv[1]
        iext.trans_elv_alav()

    # get R(V) values
    n_fit19 = len(files_fit19)
    rvs_fit19 = np.zeros((n_fit19, 2))
    for i, iext in enumerate(exts_fit19):
        irv = iext.columns["RV"]
        rvs_fit19[i, 0] = irv[0]
        rvs_fit19[i, 1] = irv[1]
        iext.trans_elv_alav()

    # get R(V) values
    n_gor21 = len(files_gor21)
    rvs_gor21 = np.zeros((n_gor21, 2))
    for i, iext in enumerate(exts_gor21):
        irv = iext.columns["RV"]
        rvs_gor21[i, 0] = irv[0]
        rvs_gor21[i, 1] = irv[1]
        iext.trans_elv_alav()

    # get R(V) values
    n_dec22 = len(files_dec22)
    rvs_dec22 = np.zeros((n_dec22, 2))
    for i, iext in enumerate(exts_dec22):
        irv = iext.columns["RV"]
        rvs_dec22[i, 0] = irv[0]
        rvs_dec22[i, 1] = irv[1]
        iext.trans_elv_alav()

    if args.rv:
        labx = "$R(V)$"
        xrange = [2.0, 6.5]
    else:
        rvs_val04[:, 0] = 1 / rvs_val04[:, 0]
        rvs_gor09[:, 0] = 1 / rvs_gor09[:, 0]
        rvs_fit19[:, 0] = 1 / rvs_fit19[:, 0]
        rvs_gor21[:, 0] = 1 / rvs_gor21[:, 0]
        rvs_dec22[:, 0] = 1 / rvs_dec22[:, 0]
        labx = "$1/R(V)$"
        xrange = [1.0 / 6.5, 1.0 / 2.0]

    fontsize = 12

    font = {"size": fontsize}

    plt.rc("font", **font)

    plt.rc("lines", linewidth=1)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("ytick.major", width=2)

    fig, fax = plt.subplots(nrows=2, ncols=4, figsize=(17, 8), sharex=True)
    ax = fax.flatten()

    repwaves = {
        "FUSE1": 0.1 * u.micron,
        "IUE1": 0.15 * u.micron,
        "IUE2": 0.2175 * u.micron,
        "IUE3": 0.3 * u.micron,
        "STIS1": 0.45 * u.micron,
        # "BAND1": 0.45 * u.micron,
        "BAND2": 2.1 * u.micron,
        "SpeX_SXD1": 2.0 * u.micron,
        "IRS1": 15.0 * u.micron,
    }

    for i, rname in enumerate(repwaves.keys()):
        if "FUSE" in rname:
            oexts = get_alav(exts_gor09, "FUSE", repwaves[rname])
            ax[i].plot(
                rvs_gor09[:, 0], oexts[:, 0], psym_gor09, fillstyle="none", label="G09"
            )
        if "STIS" in rname:
            oexts = get_alav(exts_fit19, "STIS", repwaves[rname])
            ax[i].plot(
                rvs_fit19[:, 0], oexts[:, 0], psym_fit19, fillstyle="none", label="F19"
            )
        elif "SpeX_SXD" in rname:
            oexts = get_alav(exts_dec22, "SpeX_SXD", repwaves[rname])
            ax[i].plot(
                rvs_dec22[:, 0], oexts[:, 0], psym_dec22, fillstyle="none", label="D22"
            )
        elif "IRS" in rname:
            oexts = get_alav(exts_gor21, "IRS", repwaves[rname])
            ax[i].plot(
                rvs_gor21[:, 0], oexts[:, 0], psym_gor21, fillstyle="none", label="G21"
            )
        elif "IUE" in rname:
            # oexts = get_alav(exts_val04, "IUE", repwaves[rname])
            # ax[i].plot(
            #     rvs_val04[:, 0],
            #     oexts[:, 0],
            #     psym_val04,
            #     fillstyle="none",
            #     alpha=0.5,
            #     label="V04",
            # )
            oexts = get_alav(exts_gor09, "IUE", repwaves[rname])
            ax[i].plot(
                rvs_gor09[:, 0], oexts[:, 0], psym_gor09, fillstyle="none", label="G09"
            )
            oexts = get_alav(exts_fit19, "STIS", repwaves[rname])
            ax[i].plot(
                rvs_fit19[:, 0], oexts[:, 0], psym_fit19, fillstyle="none", label="F19"
            )
            oexts = get_alav(exts_gor21, "IUE", repwaves[rname])
            ax[i].plot(
                rvs_gor21[:, 0], oexts[:, 0], psym_gor21, fillstyle="none", label="G21"
            )
            oexts = get_alav(exts_dec22, "IUE", repwaves[rname])
            ax[i].plot(
                rvs_dec22[:, 0], oexts[:, 0], psym_dec22, fillstyle="none", label="D22"
            )
        elif "BAND" in rname:
            oexts = get_alav(exts_val04, "BAND", repwaves[rname])
            ax[i].plot(
                rvs_val04[:, 0],
                oexts[:, 0],
                psym_val04,
                fillstyle="none",
                alpha=0.5,
                label="V04",
            )
            oexts = get_alav(exts_gor09, "BAND", repwaves[rname])
            ax[i].plot(
                rvs_gor09[:, 0], oexts[:, 0], psym_gor09, fillstyle="none", label="G09"
            )
            oexts = get_alav(exts_gor21, "BAND", repwaves[rname])
            ax[i].plot(
                rvs_gor21[:, 0], oexts[:, 0], psym_gor21, fillstyle="none", label="G21"
            )
            oexts = get_alav(exts_dec22, "BAND", repwaves[rname])
            ax[i].plot(
                rvs_dec22[:, 0], oexts[:, 0], psym_dec22, fillstyle="none", label="D22"
            )
        ax[i].legend(title=f"{repwaves[rname]}", ncol=2)

    fax[0, 1].set_xlim(xrange)

    # for 2nd x-axis with R(V) values
    new_ticks = np.array(
        [1.0 / 2.0, 1.0 / 3.0, 1.0 / 4.0, 1.0 / 5.0, 1.0 / 6.0]
    )
    new_ticks_labels = ["%.1f" % z for z in 1.0 / new_ticks]

    for i in range(4):
        fax[1, i].set_xlabel(labx)

        if not args.rv:
            # add 2nd x-axis with R(V) values
            tax = fax[0, i].twiny()
            tax.set_xlim(fax[0, i].get_xlim())
            tax.set_xticks(new_ticks)
            tax.set_xticklabels(new_ticks_labels)
            tax.set_xlabel(r"$R(V)$")

    for i in range(2):
        fax[i, 0].set_ylabel(r"$A(\lambda)/A(V)$")

    fig.tight_layout()

    fname = "fuv_mir_rv_rep_waves"
    if args.rv:
        fname = f"{fname}_rv"
    if args.png:
        fig.savefig(f"{fname}.png")
    elif args.pdf:
        fig.savefig(f"{fname}.pdf")
    else:
        plt.show()
