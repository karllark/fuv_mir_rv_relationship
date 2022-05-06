import numpy as np
from scipy.stats import multivariate_normal

from astropy.modeling.models import Linear1D

import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
import scipy.optimize as op


def lnlike_correlated(params, measured_vals, updated_model, cov, intinfo, x):
    """
    Compute the natural log of the likelihood that a model fits
    (x, y) data points that have correlated uncertainties given
    in the covariance matrix.  This is done by computing the line
    integral along the model evaluating the likelihood each x,y data
    point matches the model thereby getting the total likelihood
    that the data is from that model.  Should be the full solution unlike
    fitting assuming only y uncs or fitting with the orthogonal distance
    regression (ODR).

    Parameters
    ----------
    measured_vals : ndarray of length N
        Measured data values.
    updated_model : `~astropy.modeling.Model`
        Model with parameters set by the current iteration of the optimizer.
    cov : ndarray (N, 2, 2)
        2x2 covariance matrices for each (x, y) data points
    intinfo : 3 element array
        line integration info with (x min, x max, x delta) values
    x : ndarray
        Independent variable "x" on which to evaluate the model.
    """
    updated_model.slope = params[0]
    updated_model.intercept = params[1]

    modx = np.arange(intinfo[0], intinfo[1], intinfo[2])
    mody = updated_model(modx)
    pos = np.column_stack((modx, mody))
    # determine the linear distance between adjacent model points
    #    needed for line integral
    lindist = np.sqrt(np.square(modx[1:] - modx[:-1]) + np.square(mody[1:] - mody[:-1]))
    # total distance - needed for normalizing line integral
    totlength = np.sqrt((modx[-1] - modx[0]) ** 2 + (mody[-1] - mody[0]) ** 2)
    lineintegral = 0.0
    for k, xval in enumerate(x):
        # define a multivariate normal/Gaussian for each data point
        datamod = multivariate_normal([xval, measured_vals[k]], cov[k, :, :])
        # evalute the data at the model (x,y) points
        modvals = datamod.pdf(pos)
        modaves = modvals[1:] + modvals[:-1]
        tintegral = np.sum(modaves * lindist)
        # print(k, tintegral)
        if tintegral != 0.0:
            lineintegral += np.log(tintegral / totlength)
        # print(xval, measured_vals[k], lineintegral, np.sum(modaves * lindist) / totlength)
    if lineintegral == 0.0 or not np.isfinite(lineintegral):
        lineintegral = -1e20

    print(params, lineintegral, totlength)

    return lineintegral


def lnlike_correlated_quad(params, measured_vals, updated_model, cov, intinfo, x):
    """
    Compute the natural log of the likelihood that a model fits
    (x, y) data points that have correlated uncertainties given
    in the covariance matrix.  This is done by computing the line
    integral along the model evaluating the likelihood each x,y data
    point matches the model thereby getting the total likelihood
    that the data is from that model.  Should be the full solution unlike
    fitting assuming only y uncs or fitting with the orthogonal distance
    regression (ODR).

    Parameters
    ----------
    measured_vals : ndarray of length N
        Measured data values.
    updated_model : `~astropy.modeling.Model`
        Model with parameters set by the current iteration of the optimizer.
    cov : ndarray (N, 2, 2)
        2x2 covariance matrices for each (x, y) data points
    intinfo : 3 element array
        line integration info with (x min, x max, x delta) values
    x : ndarray
        Independent variable "x" on which to evaluate the model.
    """
    updated_model.c0 = params[0]
    updated_model.c1 = params[1]
    updated_model.c2 = params[2]

    modx = np.arange(intinfo[0], intinfo[1], intinfo[2])
    mody = updated_model(modx)
    pos = np.column_stack((modx, mody))
    # determine the linear distance between adjacent model points
    #    needed for line integral
    lindist = np.sqrt(np.square(modx[1:] - modx[:-1]) + np.square(mody[1:] - mody[:-1]))
    # total distance - needed for normalizing line integral
    totlength = np.sum(lindist)
    lineintegral = 0.0
    for k, xval in enumerate(x):
        # define a multivariate normal/Gaussian for each data point
        datamod = multivariate_normal([xval, measured_vals[k]], cov[k, :, :])
        # evalute the data at the model (x,y) points
        modvals = datamod.pdf(pos)
        modaves = modvals[1:] + modvals[:-1]
        tintegral = np.sum(modaves * lindist)
        # print(k, tintegral)
        if tintegral != 0.0:
            lineintegral += np.log(tintegral / totlength)
        # print(xval, measured_vals[k], lineintegral, np.sum(modaves * lindist) / totlength)
    if lineintegral == 0.0 or not np.isfinite(lineintegral):
        lineintegral = -1e20

    print(params, lineintegral, totlength)

    return lineintegral


if __name__ == "__main__":

    # define a model for a line
    line_orig = models.Linear1D(slope=1.0, intercept=0.5)

    # generate x, y data non-uniformly spaced in x
    # add noise to y measurements
    npts = 30
    np.random.seed(10)
    x = np.random.uniform(0.0, 10.0, npts)
    y = line_orig(x)

    xunc = np.absolute(np.random.normal(0.5, 2.5, npts))
    x += np.random.normal(0.0, xunc, npts)
    yunc = np.absolute(np.random.normal(0.5, 2.5, npts))
    y += np.random.normal(0.0, yunc, npts)

    covs = np.zeros((npts, 2, 2))
    for k in range(npts):
        covs[k, 0, 0] = xunc[k]
        covs[k, 0, 1] = 0.0
        covs[k, 1, 0] = 0.0
        covs[k, 1, 1] = yunc[k]

    params = [-0.5, 0.5]
    intinfo = [-5.0, 15.0, 0.1]

    def nll(*args):
        return -lnlike_correlated(*args)

    result = op.minimize(nll, params, args=(y, line_orig, covs, intinfo, x))
    nparams = result["x"]
    print(nparams)

    fitted_line = models.Linear1D(slope=nparams[0], intercept=nparams[1])

    # plot the model
    plt.figure()
    plt.errorbar(x, y, xerr=xunc, yerr=yunc, fmt="ko", label="Data")
    plt.plot(x, fitted_line(x), "k-", label="Fitted Model")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
