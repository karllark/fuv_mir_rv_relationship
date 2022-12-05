import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import erf

import matplotlib.pyplot as plt
from astropy.modeling import models
import scipy.optimize as op
import emcee


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
        Model with parameters set by the current iteration of the solver.
    cov : ndarray (N, 2, 2)
        2x2 covariance matrices for each (x, y) data points
    intinfo : 3 element array
        line integration info with (x min, x max, x delta) values
    x : ndarray
        Independent variable "x" on which to evaluate the model.
    """
    updated_model.parameters = params

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
        if tintegral != 0.0:
            # lineintegral += np.log(tintegral / totlength)
            lineintegral += np.log(tintegral)
    if lineintegral == 0.0 or not np.isfinite(lineintegral):
        lineintegral = -1e20

    return lineintegral


def lnlike_correlated_analytic(params, measured_vals, updated_model, cov, intinfo, x):
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
        Model with parameters set by the current iteration of the solver.
    cov : ndarray (N, 2, 2)
        2x2 covariance matrices for each (x, y) data points
    intinfo : 3 element array
        line integration info with (x min, x max, x delta) values
    x : ndarray
        Independent variable "x" on which to evaluate the model.
    """
    updated_model.parameters = params

    # slope, intercept = params
    slope = updated_model.slope
    intercept = updated_model.intercept

    lineint_const = np.sqrt(3. ** 2 + (intercept + 3. * slope) ** 2)

    lineintegral = 0.0
    for k in range(len(x)):
        mux = x[k]
        muy = measured_vals[k]
        sigx = np.sqrt(cov[k, 0, 0])
        sigy = np.sqrt(cov[k, 1, 1])
        rho = cov[k, 0, 1] / (sigx * sigy)
        onerhosqr = 1. - rho ** 2
        A = 1. / (2. * np.pi * sigx * sigy * np.sqrt(onerhosqr))
        B = 1. / (2. * onerhosqr * sigx ** 2)
        C = 1. / (2. * onerhosqr * sigx * sigy)
        D = 1. / (2. * onerhosqr * sigy ** 2)

        w = intercept + slope
        H1 = B * mux ** 2 + C * mux * (intercept + muy) + D * (intercept ** 2 - intercept * muy + muy ** 2)
        H2 = -2. * B * mux + C * (muy + w - intercept) + 2. * D * w * (intercept - w)
        H3 = B - C * w + D * w ** 2

        # term1 = erf((2. * H3 * 0.3 + H2)/(2. * np.sqrt(H3)))   # from + limit
        # term2 = erf((2. * H3 * -0.3 + H2)/(2. * np.sqrt(H3)))  # from -limit
        # curint = ((np.sqrt(np.pi) * np.exp((H2 ** 2 / (4. * H3)) - H1) * (term1 - term2))
        #           / (2. * np.sqrt(H3)))
        # print(curint)
        # print(H1, H2, H3)
        expval = (H2 ** 2 / (4. * H3)) - H1
        if expval < 500:
            curint = A * lineint_const * np.sqrt(np.pi) * np.exp(expval) / np.sqrt(H3)
        else:
            curint = 0.0
        # print(curint)
        if curint != 0.0 and np.isfinite(lineintegral):
            lineintegral += np.log(curint)

    if lineintegral == 0.0 or not np.isfinite(lineintegral):
        lineintegral = -1e20
        # print(A)
        # print(np.log(lineintegral))

    return lineintegral


def lnlike_correlated_fast(params, datamod, updated_model, intinfo):
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
    datamod : list
        Multivariate normal/Gaussian models for each data point
    updated_model : `~astropy.modeling.Model`
        Model with parameters set by the current iteration of the solver.
    cov : ndarray (N, 2, 2)
        2x2 covariance matrices for each (x, y) data points
    intinfo : 3 element array
        line integration info with (x min, x max, x delta) values
    """
    updated_model.parameters = params

    modx = np.arange(intinfo[0], intinfo[1], intinfo[2])
    mody = updated_model(modx)
    pos = np.column_stack((modx, mody))
    # determine the linear distance between adjacent model points
    #    needed for line integral
    lindist = np.sqrt(np.square(modx[1:] - modx[:-1]) + np.square(mody[1:] - mody[:-1]))
    # total distance - needed for normalizing line integral
    totlength = np.sum(lindist)
    lineintegral = 0.0
    for cdatamod in datamod:
        # evalute the data at the model (x,y) points
        modvals = cdatamod.pdf(pos)
        modaves = modvals[1:] + modvals[:-1]
        # integrate
        tintegral = np.sum(modaves * lindist)
        if tintegral != 0.0:
            lineintegral += np.log(tintegral / totlength)
    if lineintegral == 0.0 or not np.isfinite(lineintegral):
        lineintegral = -1e20

    return lineintegral


def fit_2Dcorrelated(x, y, covs, fit_model, intinfo):
    """
    Do standard optimization fitting with correlated lnlike function.
    """

    def nll(*args):
        return -lnlike_correlated(*args)

    params = fit_model.parameters
    # print("start:", params)
    result = op.minimize(nll, params, args=(y, fit_model, covs, intinfo, x))

    fit_model.parameters = result["x"]
    fit_model.result = result
    # print("end:", fit_model.parameters)
    return fit_model


def fit_2Dcorrelated_fast(x, y, covs, fit_model, intinfo):
    """
    Do standard optimization fitting with correlated lnlike function.
    """

    datamod = []
    for k in range(len(x)):
        datamod.append(multivariate_normal([x[k], y[k]], covs[k, :, :]))

    def nll(*args):
        return -lnlike_correlated_fast(*args)

    params = fit_model.parameters
    # print("start:", params)
    # result = op.minimize(nll, params, args=(datamod, fit_model, intinfo))
    result = op.minimize(nll, params, args=(datamod, fit_model, intinfo))

    fit_model.parameters = result["x"]
    fit_model.result = result
    # print("end:", fit_model.parameters)
    return fit_model


def fit_2Dcorrelated_analytic_fast(x, y, covs, fit_model, intinfo):
    """
    Do standard optimization fitting with correlated lnlike function.
    """

    datamod = []
    for k in range(len(x)):
        datamod.append(multivariate_normal([x[k], y[k]], covs[k, :, :]))

    def nll(*args):
        return -lnlike_correlated_analytic(*args)

    params = fit_model.parameters
    # print("start:", params)
    # result = op.minimize(nll, params, args=(datamod, fit_model, intinfo))
    # result = op.minimize(nll, params, args=(datamod, fit_model, intinfo))
    result = op.minimize(nll, params, args=(y, fit_model, covs, intinfo, x))

    fit_model.parameters = result["x"]
    fit_model.result = result
    # print("end:", fit_model.parameters)
    return fit_model


def fit_2Dcorrelated_emcee(x, y, covs, fit_model, intinfo, nsteps=100, progress=False,
                           sfilename=None):
    """
    Do emcee based sampling with correlated lnlike function.
    """
    datamod = []
    for k in range(len(x)):
        datamod.append(multivariate_normal([x[k], y[k]], covs[k, :, :]))

    ndim = len(fit_model.parameters)
    nwalkers = 2 * ndim
    pos = fit_model.parameters + 1e-3 * np.random.randn(nwalkers, ndim)

    if sfilename is not None:
        backend = emcee.backends.HDFBackend(sfilename)
        backend.reset(nwalkers, ndim)
    else:
        backend = None

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, lnlike_correlated_fast, args=(datamod, fit_model, intinfo),
        # nwalkers, ndim, lnlike_correlated_analytic, args=(y, fit_model, covs, intinfo, x),
        backend=backend
    )
    sampler.run_mcmc(pos, nsteps, progress=progress)

    fit_model.sampler = sampler

    return fit_model


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
    # print(nparams)

    fitted_line = models.Linear1D(slope=nparams[0], intercept=nparams[1])

    # plot the model
    plt.figure()
    plt.errorbar(x, y, xerr=xunc, yerr=yunc, fmt="ko", label="Data")
    plt.plot(x, fitted_line(x), "k-", label="Fitted Model")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
