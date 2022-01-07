import copy
import warnings
import numpy as np
import astropy.units as u

from measure_extinction.merge_obsspec import _wavegrid


def rebin_extdata(ext, source, waverange, resolution):
    """
    Rebin the source extinction curve contained in ExtData.

    Parameters
    ----------
    ext : measure_extinction ExtData
        Object with full extinction information
    source : str
        source of extinction (i.e. "IUE", "IRS")
    waverange : [float, float]
        Min/max of wavelength range
    resolution : float
        Spectral resolution of rebinned extinction curve

    Returns
    -------
    measure_extinction ExtData
        Object with source extinciton curve rebinned

    """
    if source == "BAND":
        raise ValueError("BAND extinction cannot be rebinned")

    if source not in ext.exts.keys():
        warnings.warn(f"{source} extinction not present")
        return ext

    # setup wavelength grid
    full_wave, full_wave_min, full_wave_max = _wavegrid(
        resolution, waverange.to(u.micron).value
    )
    n_waves = len(full_wave)

    # setup the output ExtData
    outext = copy.deepcopy(ext)

    outext.waves[source] = full_wave * u.micron
    outext.exts[source] = np.zeros((n_waves), dtype=float)
    outext.uncs[source] = np.zeros((n_waves), dtype=float)
    outext.npts[source] = np.zeros((n_waves), dtype=int)

    owaves = ext.waves[source].to(u.micron).value
    for k in range(n_waves):
        (indxs,) = np.where(
            (owaves >= full_wave_min[k])
            & (owaves < full_wave_max[k])
            & (ext.uncs[source] > 0.0)
        )
        if len(indxs) > 0:
            weights = 1.0 / np.square(ext.uncs[source][indxs])
            sweights = np.sum(weights)
            outext.exts[source][k] = np.sum(weights * ext.exts[source][indxs]) / sweights
            outext.uncs[source][k] = 1.0 / np.sqrt(sweights)
            outext.npts[source][k] = np.sum(ext.npts[source][indxs])

    return outext
