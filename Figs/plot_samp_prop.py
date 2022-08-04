import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np

from measure_extinction.extdata import ExtData


def plot_props(ax, avs, rvs, psym, label):
    ax.plot(
        1 / rvs[:, 0] - (1 / 3.1),
        avs[:, 0],
        psym,
        fillstyle="none",
        label=label,
        alpha=0.75,
    )
    print(label, len(avs[:, 0]))
    print("AV", np.min(avs[:, 0]), np.max(avs[:, 0]))
    print("RV", np.min(rvs[:, 0]), np.max(rvs[:, 0]))
    yerr = (1 / rvs[:, 0]) * (rvs[:, 1] / rvs[:, 0])
    # ax.errorbar(
    #     1 / rvs[:, 0] - 1 / 3.1,
    #     avs[:, 0],
    #     xerr=avs[:, 1],
    #     yerr=yerr,
    #     fmt=psym,
    #     fillstyle="none",
    #     label=label,
    #     alpha=0.2,
    # )


def hname(tname):
    """
    Make all the star names have the right number of 0 characters
    """
    nname = tname
    if tname[0:2] == "hd":
        num = tname.split("hd")[1]
        if len(num) == 6:
            nname = tname
        elif len(num) == 5:
            nname = f"hd0{num}"
        elif len(num) == 4:
            nname = f"hd00{num}"
        elif len(num) == 3:
            nname = f"hd000{num}"
        elif len(num) == 2:
            nname = f"hd0000{num}"
        else:
            print(tname)
            print("hname error")
            exit()
    else:
        nname = tname

    return nname


def check_overlap(samp1, samp2):
    """
    Compute the overlap between two samples
    """
    nmatch = 0
    for cname in samp2:
        if cname in samp1:
            nmatch += 1
    return nmatch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    names_gor09 = []
    rvs_gor09 = np.zeros((n_gor09, 2))
    avs_gor09 = np.zeros((n_gor09, 2))
    for i, iext in enumerate(exts_gor09):
        names_gor09.append(files_gor09[i].split("_")[1].lower())

        av = iext.columns["AV"]
        avs_gor09[i, 0] = av[0]
        avs_gor09[i, 1] = av[1]

        irv = iext.columns["RV"]
        rvs_gor09[i, 0] = irv[0]
        rvs_gor09[i, 1] = irv[1]

    # n_val04 = len(files_val04)
    # rvs_val04 = np.zeros((n_val04, 2))
    # avs_val04 = np.zeros((n_val04, 2))
    # for i, iext in enumerate(exts_val04):
    #     av = iext.columns["AV"]
    #     avs_val04[i, 0] = av[0]
    #     avs_val04[i, 1] = av[1]
    #
    #     irv = iext.columns["RV"]
    #     rvs_val04[i, 0] = irv[0]
    #     rvs_val04[i, 1] = irv[1]

    # get R(V) values
    n_fit19 = len(files_fit19)
    names_fit19 = []
    rvs_fit19 = np.zeros((n_fit19, 2))
    avs_fit19 = np.zeros((n_fit19, 2))
    for i, iext in enumerate(exts_fit19):
        names_fit19.append(hname(files_fit19[i].split("_")[1].lower()))

        av = iext.columns["AV"]
        avs_fit19[i, 0] = av[0]
        avs_fit19[i, 1] = av[1]

        irv = iext.columns["RV"]
        rvs_fit19[i, 0] = irv[0]
        rvs_fit19[i, 1] = irv[1]

    # get R(V) values
    n_gor21 = len(files_gor21)
    names_gor21 = []
    rvs_gor21 = np.zeros((n_gor21, 2))
    avs_gor21 = np.zeros((n_gor21, 2))
    for i, iext in enumerate(exts_gor21):
        names_gor21.append(files_gor21[i].split("_")[1].lower())

        av = iext.columns["AV"]
        avs_gor21[i, 0] = av[0]
        avs_gor21[i, 1] = 0.5 * (av[1] + av[2])

        irv = iext.columns["RV"]
        rvs_gor21[i, 0] = irv[0]
        rvs_gor21[i, 1] = irv[1]

    # get R(V) values
    n_dec22 = len(files_dec22)
    names_dec22 = []
    rvs_dec22 = np.zeros((n_dec22, 2))
    avs_dec22 = np.zeros((n_dec22, 2))
    for i, iext in enumerate(exts_dec22):
        names_dec22.append(files_dec22[i].split("/")[2].split("_")[0])

        av = iext.columns["AV"]
        avs_dec22[i, 0] = av[0]
        avs_dec22[i, 1] = 0.5 * (av[1] + av[2])

        irv = iext.columns["RV"]
        rvs_dec22[i, 0] = irv[0]
        rvs_dec22[i, 1] = 0.5 * (irv[1] + irv[2])

    fontsize = 14

    font = {"size": fontsize}

    plt.rc("font", **font)

    plt.rc("lines", linewidth=2)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("ytick.major", width=2)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5.5))

    all_tags = ["GCC09", "F19", "G21", "D22"]
    all_names = [names_gor09, names_fit19, names_gor21, names_dec22]

    plot_props(ax, avs_gor09, rvs_gor09, psym_gor09, "G09")
    for i, csamp in enumerate(all_names):
        print(all_tags[i], check_overlap(names_gor09, csamp))
    # plot_props(ax, avs_val04, rvs_val04, psym_val04, "V04")
    plot_props(ax, avs_fit19, rvs_fit19, psym_fit19, "F19")
    for i, csamp in enumerate(all_names):
        print(all_tags[i], check_overlap(names_fit19, csamp))
    plot_props(ax, avs_gor21, rvs_gor21, psym_gor21, "G21")
    for i, csamp in enumerate(all_names):
        print(all_tags[i], check_overlap(names_gor21, csamp))
    plot_props(ax, avs_dec22, rvs_dec22, psym_dec22, "D22")
    for i, csamp in enumerate(all_names):
        print(all_tags[i], check_overlap(names_dec22, csamp))

    ax.set_ylabel(r"$A(V)$")
    ax.set_xlabel(r"$1 / R(V) - 1 / 3.1$")

    xrange = np.array([1.0 / 6.5, 1.0 / 2.25]) - 1 / 3.1
    ax.set_xlim(xrange)
    ax.set_ylim([0.0, 6.0])

    # for 2nd x-axis with R(V) values
    axis_rvs = np.array([2.3, 2.5, 3.0, 4.0, 5.0, 6.0])
    new_ticks = 1 / axis_rvs - (1 / 3.1)
    new_ticks_labels = ["%.1f" % z for z in axis_rvs]
    tax = ax.twiny()
    tax.set_xlim(ax.get_xlim())
    tax.set_xticks(new_ticks)
    tax.set_xticklabels(new_ticks_labels)
    tax.set_xlabel(r"$R(V)$")

    ax.legend()

    fig.tight_layout()

    fname = "fuv_mir_samp_prop"
    if args.png:
        fig.savefig(f"{fname}.png")
    elif args.pdf:
        fig.savefig(f"{fname}.pdf")
    else:
        plt.show()
