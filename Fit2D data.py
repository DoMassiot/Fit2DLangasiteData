# 'Git2D dataset.py' - 2022 03 25
# fit of 2D microscopy data with gaussian lineshapes
# by Dominique Massiot - CEMHTI-CNRS UPR3079, Orléans France
#
# used in the manuscript "Stabilisation of the trigonal langasite structure in Ca3Ga2-2xZnxGe4+xO14 (0 ≤ x ≤ 1) with partial ordering of three isoelectronic cations characterised by a multi-technique approach"
# by Haytem Bazzaoui, Cécile Genevois, Dominique Massiot, Vincent Sarou-Kanian, Emmanuel Veron, Sébastien Chenu, Přemysl Beran, Michael J. Pitcher and Mathieu Allix
# submitted to ??
# preprint available at ??
# scripts available in GitHub : https://github.com/DoMassiot/Fit2DLangasiteData/
#
# the fitting routine was derived from
# https://scipython.com/blog/non-linear-least-squares-fitting-of-a-two-dimensional-data/
#
# the "make zone datasets.py" script has to be executed piror to this script to generate the zone datasets

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import csv
import os
import sys
from datetime import date

angstromPerPoint = 0.071  # point to angstrom calibration
angstromCalibrated = True  # True if calibration in Angstrom
zone = 0  # can be 0 or 1 to select the Zone to be fitted
method = 'lm'  # 'trf'          # method for the fit routine
do_plot_all = False  # True for plotting all graphs - False if only results
do_save_plot = True  # save the jpg images of the results

# Default parameters to start with
amp = 32000.0  # default amplitude of 2D gaussian lines
width_ini = 9  # default value for the gaussian width in point
nlines = 0  # number of lines to be fitted
nparperline = 0  # number of fitted parameters per line
nparglobal = 0  # number of global parametesr
guess_prms = np.array([])  # initial guess parameters
legSites = None  # site legends
Z = np.array([0])  # data
avg_sites_gega = np.array([])  # storage of average Ge and Ga fitting parameters of sites 1a & 3f

cmap = ""
endname = ""
dx = 0
dy = 0

def dir_test_exists(d):
    if not os.path.exists(d):
        print(f"\tdirectory {d} did not exist - it has been created")
        os.mkdir(d)
    return d

current_directory = os.getcwd()  # current directory
data_directory = "zone_data" + os.sep  # directory where data are found
fit_results_directory = "fit_results" + os.sep  # directory for saving the figures
dir_test_exists(fit_results_directory)

def check_platform():
    # checks for the operating system and build roots for filenames
    global data_directory, fit_results_directory, current_directory
    print(f"{os.name} running on {sys.platform} - separator : '{os.sep}'")
    print("Current working directory:", os.getcwd())
    print(f"data directory : '{data_directory}'")
    print(f"fit results directory : '{fit_results_directory}'")


# routines for plotting data
def axis_label():
    return "Angstrom" if angstromCalibrated else "point"


def angstrom(v):
    return v * angstromPerPoint if angstromCalibrated else v


def angstrom_x(xpoint):
    return (xpoint - (X.max() - X.min()) / 2.0) * angstromPerPoint if angstromCalibrated else xpoint


def angstrom_y(ypoint):
    return (ypoint - (Y.max() - Y.min()) / 2.0) * angstromPerPoint if angstromCalibrated else ypoint


def angstrom_scale():  # handles both values or tables...
    return angstrom_x(X), angstrom_y(Y)


def plot_data(data, titre=""):
    fig = plt.figure()
    # 3D
    # ax = fig.add_subplot(121)
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, data, cmap=cmap)
    ax.set_zlim(0, np.max(data) + 2)
    plt.title(nuclei if titre == "" else f"{nuclei} {titre}")
    # contour
    bx = fig.add_subplot(122)
    bx.imshow(data, origin='lower', cmap=cmap,
              extent=(X.min(), X.max(), Y.min(), Y.max()))
    bx.contour(X, Y, data, colors='w')
    # finalize
    plt.title(nuclei if titre == "" else f"{nuclei} {titre}")
    plt.show()


def plot3d(data, titre=""):
    xloc, yloc = angstrom_scale()
    # Plot the 3D figure of the fitted function
    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(xloc, yloc, data, cmap=cmap)
    ax.set_zlim(0, np.max(data) + 2)
    ax.set_xlabel(axis_label())
    ax.set_ylabel(axis_label())
    plt.title(nuclei if titre == "" else f"{nuclei} {titre}")
    plt.colorbar(surf, ax=ax)
    plt.show()


# ContourPlot
def contour_plot(data, titre="", save_image=False):
    xloc, yloc = angstrom_scale()
    # Plot the test data as a 2D image and the fit as overlaid contours.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(data, origin='lower', cmap=cmap,
              extent=(xloc.min(), xloc.max(), yloc.min(), yloc.max()))
    ax.contour(xloc, yloc, data, colors='w')
    ax.set_xlabel(axis_label())
    ax.set_ylabel(axis_label())
    title = nuclei if titre == "" else f"{nuclei} {titre}"
    plt.title(title)
    plt.show()
    if save_image:
        plt.savefig(f"{fit_results_directory}{title} {endname}.png")


# Plot the 3D figure of the fitted function and the residuals.
def plot_fit_3d(data, fit, groffset, save_image=True, titre=""):
    xloc, yloc = angstrom_scale()

    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(xloc, yloc, fit, cmap=cmap)
    ax.contourf(xloc, yloc, data - fit, zdir='z', offset=-groffset, cmap=cmap)
    ax.set_zlim(-groffset, np.max(fit))
    ax.set_xlabel(axis_label())
    ax.set_ylabel(axis_label())

    plt.colorbar(surf, ax=ax)
    title = f"{nuclei} fit3D" if titre == "" else f"{nuclei} {titre}"
    plt.title(title)
    if save_image:
        plt.savefig(f"{fit_results_directory}{title} {endname}.png")
    plt.show()


# Plot contour density plot of data + contour of model
def plot_fit_contour(data, fit=None, save_image=True, titre="", zmax=None, subplot=111):
    xloc, yloc = angstrom_scale()
    # Plot the test data as a 2D image and the fit as overlaid contours.
    fig = plt.figure()
    ax = fig.add_subplot(subplot)
    img = ax.imshow(data, origin='lower', cmap=cmap,
                    extent=(xloc.min(), xloc.max(), yloc.min(), yloc.max()), vmin=data.min(),
                    vmax=zmax if zmax else data.max())
    plt.colorbar(img)
    hasFit = type(fit) == np.ndarray
    if hasFit:
        ax.contour(xloc, yloc, fit, colors='w')

    ax.set_xlabel(axis_label())
    ax.set_ylabel(axis_label())
    title = f"{nuclei} fit" if titre == "" else f"{nuclei} {titre}"
    plt.title(title)
    if save_image:
        plt.savefig(f"{fit_results_directory}{title} {endname}.png")
        if hasFit:
            np.savetxt(f"{fit_results_directory}{title} {endname} data.csv", data, delimiter=';')
            np.savetxt(f"{fit_results_directory}{title} {endname} fit.csv", fit, delimiter=';')
    if subplot == 111:
        plt.show()


# retrieve data from csv files
def read_csv_data(nucName):
    global X, Y

    def make_save_name(nucleus):
        return f"{data_directory}{nucleus} {endname}.csv"

    def split_row(row):
        return [int(i) for i in row[0].split(",")]

    fname = make_save_name(nucName)
    print(f"filename : '{fname}'")
    with open(fname, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        t = [split_row(row) for row in spamreader]
        x = t[0]
        y = t[1]
        t = t[2:]  # t is now the 2D data table
        ly = len(t)
        lx = len(t[0])
        minval = min([v for sublist in t for v in sublist])
        maxval = max([v for sublist in t for v in sublist])
        print(f"table '{fname}' is {lx}x{ly}, min: {minval}, max:{maxval}")
        X, Y = np.meshgrid(x, y)
    return np.array(t)


# selection of the Zone data
def init_zone(_zone):
    global zone, endname, cmap, dx, dy
    zone = _zone
    if zone == 0:
        endname = "x(40,104)-y(50,112)"  # "reduced"
        cmap = "viridis"
        dx = 0
        dy = 0
    else:
        endname = "x(30,94)-y(8,70)"  # "range(30, 94)-range(8, 70)"
        cmap = "plasma"
        dx = 10
        dy = 42
    with open(f"{fit_results_directory} report {endname}.txt", 'w') as f:
        f.write(f"Fit of Zone#{zone} '{endname}' - date:{date.today()}\n----------------------------\n")


# builds the mask of zeros in the dataset (used in the fit routine)
def build_mask(data):
    m = np.zeros(X.shape)
    for c in range(m.shape[1]):
        for r in range(m.shape[0]):
            m[r][c] = 1 if data[r][c] else 0
    return m


# Our function to fit is going to be a sum of two-dimensional Gaussians
def gaussian_1d(x, alpha, x0, amp):
    alpha /= 2 * np.sqrt(np.log(2))
    return amp * np.exp(-((x - x0) / alpha) ** 2)


def gaussian(x, y, alpha, x0, y0, amp):
    alpha /= 2 * np.sqrt(np.log(2))
    return amp * np.exp(-((x - x0) / alpha) ** 2 - ((y - y0) / alpha) ** 2)


def check_gaussian():
    plt.figure()
    xrange = np.arange(-5, 5, 0.05)
    fwmh = 2
    g = gaussian_1d(xrange, fwmh, 0, 1)
    step = [(0 if x < -1 or x > 1 else 0.5) for x in xrange]
    plt.plot(xrange, g)
    plt.plot(xrange, step)
    plt.title(f"Gaussian of width {fwmh}")
    plt.show()


def add_csv_data(nucleus):
    global Z, amp
    if Z.max() > 0:
        data = Z + read_csv_data(nucleus)
    else:
        data = read_csv_data(nucleus)
    amp = data.max() - data.min()
    return data


legSitesCa = ["Ca", "Ca", "Ca", "Ca", "Ca", "Ca"]
legSites2d = ["2d" for i in range(6)]
legSites1a3f = ["1a"] + ["3f" for i in range(3)]
legSitesAll = legSitesCa + legSites2d + legSites1a3f


def sitesCa():
    return [  # 6 sites Ca seulement
        54.0 - dx, 76.0 - dy, amp,
        66.0 - dx, 58.0 - dy, amp,
        85.0 - dx, 70.0 - dy, amp,
        95.0 - dx, 89.0 - dy, amp,
        75.0 - dx, 99.0 - dy, amp,
        52.0 - dx, 99.0 - dy, amp,
        10000.0, width_ini]


legSitesGe = legSites2d + legSites1a3f


def sitesGe():
    return [
        # 6 sites 2f
        54 - dx, 64 - dy, amp,
        80 - dx, 58 - dy, amp,
        95 - dx, 76 - dy, amp,
        88 - dx, 99 - dy, amp,
        62 - dx, 105 - dy, amp,
        46 - dx, 87 - dy, amp,
        #
        71 - dx, 80 - dy, amp,
        69 - dx, 72 - dy, amp,
        80 - dx, 85 - dy, amp,
        65 - dx, 87 - dy, amp,
        10000.0, width_ini]


def sitesGaZn():
    return [
        #
        71 - dx, 80 - dy, amp,
        69 - dx, 72 - dy, amp,
        80 - dx, 85 - dy, amp,
        65 - dx, 87 - dy, amp,
        10000.0, width_ini]


def sitesCaGe():
    return [
        # site Ca
        54.0 - dx, 76.0 - dy, amp,
        66.0 - dx, 58.0 - dy, amp,
        85.0 - dx, 70.0 - dy, amp,
        95.0 - dx, 89.0 - dy, amp,
        75.0 - dx, 99.0 - dy, amp,
        52.0 - dx, 99.0 - dy, amp,
        # 6 sites 2f
        54 - dx, 64 - dy, amp,
        80 - dx, 58 - dy, amp,
        95 - dx, 76 - dy, amp,
        88 - dx, 99 - dy, amp,
        62 - dx, 105 - dy, amp,
        46 - dx, 87 - dy, amp,
        # site 1a
        71 - dx, 80 - dy, amp,
        # sites 3f
        69 - dx, 72 - dy, amp,
        80 - dx, 85 - dy, amp,
        65 - dx, 87 - dy, amp,
        10000.0, width_ini]


def init_nuclei(nucnames):  # Initial guesses to the fit parameters.
    global nlines, nparperline, nparglobal, guess_prms
    global Z, X, Y, legSites

    Z = np.array([0])
    if "Ca" in nucnames:
        Z = add_csv_data("Ca")
        nlines = 6
        nparperline = 3
        nparglobal = 2
        guess_prms = sitesCa()
        legSites = legSitesCa
    if "Ga" in nucnames:
        Z = add_csv_data("Ga")
        nlines = 4
        nparperline = 3
        nparglobal = 2
        guess_prms = sitesGaZn()
        legSites = legSites1a3f
    if "Zn" in nucnames:
        Z = add_csv_data("Zn")
        nlines = 4
        nparperline = 3
        nparglobal = 2
        guess_prms = sitesGaZn()
        legSites = legSites1a3f
    if "Ge" in nucnames:
        Z = add_csv_data("Ge")
        nlines = 10
        nparperline = 3
        nparglobal = 2
        guess_prms = sitesGe()
        legSites = legSites2d + legSites1a3f
    if "Ca" in nucnames and "Ge" in nucnames:
        nlines = 16
        nparperline = 3
        nparglobal = 2
        guess_prms = sitesCaGe()
        legSites = legSitesAll


# The function to be fit is Z.
def model(parms, Offset=True):
    offset = parms[-2] if Offset else 0
    mod = np.zeros(X.shape)
    for i in range(nlines):
        mod += gaussian(X, Y, parms[-1], *parms[i * nparperline:(i + 1) * nparperline])
    return (mod + offset) * mask


def report_parameters(parms, legend="legend ?", tt=1.0, perr=None, rms=None):
    has_errors = "ndarray" in f"{type(perr)}" or "list" in f"{type(perr)}"
    report = f'{legend} for {nuclei}:\n'
    report += f"values in Angstrom with {angstromPerPoint} A/point\n" if angstromCalibrated else "values in points...\n"
    np.set_printoptions(precision=2, suppress=True)
    sumAmp = sum([parms[i * nparperline + 2] for i in range(nlines)])
    for i in range(nlines):
        report += f"{legSites[i]}\t"
        #        for n in range(nparperline-1):
        #            report += f"{angstrom(parms[i*nparperline+n]):0.2f}\t"
        report += f"{angstrom_x(parms[i * nparperline]):0.2f}\t"  # pos X
        if has_errors and not perr[i * nparperline] == 0.0:
            report += f"+/-{angstrom(perr[i * nparperline]):0.2f}\t"  # error on pos X

        report += f"{angstrom_y(parms[i * nparperline + 1]):0.2f}\t"  # pos Y
        if has_errors and not perr[i * nparperline + 1] == 0.0:
            report += f"+/-{angstrom(perr[i * nparperline + 1]):0.2f}\t"  # error on pos Y

        report += f"{parms[i * nparperline + 2]:0.2f}\t"  # amplitude
        if has_errors and not perr[i * nparperline + 2] == 0.0:
            report += f"+/-{perr[i * nparperline + 2]:0.2f}\t"

        report += f"{parms[i * nparperline + 2] / sumAmp * 100:0.2f}%"
        if has_errors and not perr[i * nparperline + 2] == 0.0:
            report += f"\t+/-{perr[i * nparperline + 2] / sumAmp * 100:0.2f}"
        report += "\n"

    report += f"Width\t{angstrom(parms[-1]):0.2f}"
    if has_errors and not perr[-1] == 0.0:
        report += f"\t+/-{angstrom(perr[-1]):0.2f}"
    report += "\n"

    report += f"Offset\t{parms[-2]:0.2f}"
    if has_errors and not perr[-2] == 0.0:
        report += f"\t+/-{perr[-2]:0.2f}"
    report += "\n"
    if rms:
        report += f'RMS residual :\t{rms:0.2f}\n'
    if has_errors:
        with open(f"{fit_results_directory} report {endname}.txt", 'a') as f:
            f.write(report + "\n\n")
        report += f"error matrix:\n{perr}\n"
    return report


def report_fit_all(popt, perr, plotAll=True, ):
    fit = model(popt)
    print(report_parameters(popt, 'Fitted parameters', perr=perr, rms=np.sqrt(np.mean((Z - fit) ** 2))))

    groffset = popt[-2]
    fitcorrected = (fit - popt[-2]) * mask  # removing offset ;-)
    Zcorrected = (Z - popt[-2]) * mask  # removing offset ;-)

    if plotAll:
        plot_fit_3d(Zcorrected, fitcorrected, groffset, save_image=True)
    plot_fit_contour(Zcorrected, fitcorrected, save_image=True)  # , subplot=211)
    plot_fit_contour(Z - fit, save_image=True, titre="difference", zmax=Zcorrected.max())  # , subplot=212)
    plt.show()
    return popt, perr


# This is the callable that is passed to curve_fit. M is a (2,N) array
# where N is the total number of data points in Z, which will be ravelled
# to one dimension.
def do_fit(Nuclei, plotAll=True, saveImage=True):
    def _gaussian(M, *args):
        x, y = M
        arr = np.zeros(x.shape)
        for i in range(nlines):
            arr += gaussian(x, y, args[-1], *args[i * nparperline:i * nparperline + nparperline])
        return (arr + args[-2]) * mask.ravel()

    global nuclei, Z, mask
    nuclei = Nuclei
    init_nuclei(nuclei)
    # plotData(Z,"data")
    if plotAll:
        contour_plot(Z, "data")
    # plot3D(Z, "data")
    mask = build_mask(Z)
    # plot3D(mask, "mask")

    if plotAll:
        print(report_parameters(guess_prms, "starting parameters [guess]"))
        plot3d(model(guess_prms, False), "starting parameters [guess]")

    # We need to ravel the meshgrids of X, Y points to a pair of 1-D arrays.
    xdata = np.vstack((X.ravel(), Y.ravel()))
    # Do the fit, using our custom _gaussian function which understands our
    # flattened (ravelled) ordering of the data points.
    popt, pcov = curve_fit(_gaussian, xdata, Z.ravel(), guess_prms, method=method)
    # computing standard errors from cov matrix : np.sqrt(np.diag(pcov))
    # see SciPy documentation
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

    perr = np.sqrt(np.diag(pcov))
    return report_fit_all(popt, perr, plotAll)


def get_avg_gega(opt):
    if not ("Ge" in opt.keys() and "Ga" in opt.keys()):
        print(f"no existing data for Ge and Ga... cannot average")
        return np.array([])
    avg = (opt["Ga"][0] + opt["Ge"][0][18:]) / 2.0
    print("average positions and width from Ge and Ga 1a & 3f sites")
    print("--------------------------------------------------------")
    print("site\tx\ty")
    for i in range(4):
        print(f"{legSites1a3f[i]}\t{avg[i * 3]:10.2f}\t{avg[i * 3 + 1]:10.2f}")
    print(f"averaged width : {avg[-1]:10.2f} point {angstromPerPoint * avg[-1]:10.2f} Angstrom")
    return avg


def init_guess_zn(do_report=True):
    # initialize parameters for Zn sites - taking average results from Ge and Ga if they exist
    global nuclei

    nuclei = "Zn"
    init_nuclei(nuclei)
    if avg_sites_gega.size > 1:
        guess_prmsZn = avg_sites_gega.tolist()
        for n in range(4):
            guess_prmsZn[3 * n + 2] = amp
    else:
        if zone == 0:
            guess_prmsZn = [
                71.4, 81.4, amp,
                68.8, 71.5, amp,
                81.7, 85.3, amp,
                63.4, 88.0, amp,
                15000, width_ini
            ]
        else:
            guess_prmsZn = [
                62.5, 40.1, amp,
                59.3, 30.0, amp,
                71.8, 43.5, amp,
                54.8, 47.1, amp,
                15000, width_ini
            ]
    if do_report:
        print(report_parameters(guess_prmsZn, "Zn Guess parameters"))
    return guess_prmsZn


def doFitZnWidth(plot_all=True):
    def _gaussian(M, *args):
        x, y = M
        arr = np.zeros(x.shape)
        for i in range(nlines):
            arr += gaussian(x, y, args[-1], parmsZn[i * nparperline], parmsZn[i * nparperline + 1],
                            *args[i:i + 1])  # amplitude for each line and common width
        return (arr + args[-2]) * mask.ravel()

    global nuclei, Z, mask
    parmsZn = init_guess_zn()
    # plotData(Z,"data")
    if plot_all:
        contour_plot(Z, "data")
    # plot3D(Z, "data")
    mask = build_mask(Z)
    # plot3D(mask, "mask")

    if plot_all:
        print(report_parameters(parmsZn, "starting parameters [guess]"))
        plot3d(model(parmsZn, False), "starting parameters [guess]")

    # We need to ravel the meshgrids of X, Y points to a pair of 1-D arrays.
    xdata = np.vstack((X.ravel(), Y.ravel()))
    # Do the fit, using our custom _gaussian function which understands our
    # flattened (ravelled) ordering of the data points.
    guess = [parmsZn[i * nparperline + 2] for i in range(nlines)] + parmsZn[-2:]
    tmpopt, pcov = curve_fit(_gaussian, xdata, Z.ravel(), guess, method=method)
    tmperr = np.sqrt(np.diag(pcov))

    # rebuild a complete set of parameters
    popt = parmsZn
    perr = [0 for i in range(len(parmsZn))]
    for i in range(nlines):  # update amplitudes
        popt[i * nparperline + 2] = tmpopt[i]
        perr[i * nparperline + 2] = tmperr[i]

    # update offset and width
    popt[-1] = tmpopt[-1]
    popt[-2] = tmpopt[-2]
    perr[-1] = tmperr[-1]
    perr[-2] = tmperr[-2]
    return report_fit_all(np.array(popt), np.array(perr), plotAll=plot_all)


def doFitZnNoWidth(plotAll=True):
    def _gaussian(M, *args):
        x, y = M
        arr = np.zeros(x.shape)
        for i in range(nlines):
            arr += gaussian(x, y, parmsZn[-1], parmsZn[i * nparperline], parmsZn[i * nparperline + 1],
                            *args[i:i + 1])  # amplitude for each line and fixed common width
        #            arr += gaussian(x, y, parmsZn[-1], parmsZn[i * nparperline], parmsZn[i * nparperline + 1],
        #                            *args[i:i + 1])  # only amplitude is from args...
        return (arr + args[-1]) * mask.ravel()  # offset is -1

    global nuclei, Z, mask
    # plotData(Z,"data")
    parmsZn = init_guess_zn()
    if plotAll:
        contour_plot(Z, "data")
    # plot3D(Z, "data")
    mask = build_mask(Z)
    # plot3D(mask, "mask")

    if plotAll:
        plot3d(model(parmsZn, False), "starting parameters [guess]")

    # We need to ravel the meshgrids of X, Y points to a pair of 1-D arrays.
    xdata = np.vstack((X.ravel(), Y.ravel()))
    # Do the fit, using our custom _gaussian function which understands our
    # flattened (ravelled) ordering of the data points.
    guess = [parmsZn[i * nparperline + 2] for i in range(nlines)] + parmsZn[-2:-1]
    tmpopt, pcov = curve_fit(_gaussian, xdata, Z.ravel(), guess, method=method)
    tmperr = np.sqrt(np.diag(pcov))

    # rebuild the complete set of parameters
    popt = parmsZn
    perr = [0 for i in range(len(parmsZn))]
    for i in range(nlines):  # update amplitudes
        popt[i * nparperline + 2] = tmpopt[i]
        perr[i * nparperline + 2] = tmperr[i]

    # update offset [-1] of rmppopt and [-2] of popt
    popt[-2] = tmpopt[-1]
    perr[-2] = tmperr[-1]

    return report_fit_all(np.array(popt), np.array(perr), plotAll=plotAll)


def do_fit_zone(_zone):
    global avg_sites_gega
    init_zone(_zone)
    optimized = {}
    # optimize for amplitudes, positions, common width and offset
    optimized = {nuc: do_fit(nuc, do_plot_all, do_save_plot) for nuc in ["Ca", "Ge", "Ga"]}
    # take average positions and width for 1a & 3f sites from Ge and Ga fits if exist
    avg_sites_gega = get_avg_gega(optimized)

    # optimize Zn with variable width
    # optimized["Zn_width"] = doFitZnWidth(False)

    # optimize Zn with fixed width
    optimized["Zn"] = doFitZnNoWidth(True)

    return optimized


# check_gaussian()   # this is to check that we properly use fwhm

check_platform()
opt_all = {0: do_fit_zone(0), 1: do_fit_zone(1)}

print("\n\nfinal results")
print("-------------")
for kopt in opt_all.keys():
    for k in opt_all[kopt].keys():
        print(f"Zone{kopt}-{k}: {opt_all[kopt][k]}")
