import numpy as np

from astropy.modeling import Fittable1DModel, Parameter

# from dust_extinction.shapes import G21
from dust_extinction.helpers import _get_x_in_wavenumbers, _test_valid_x_range
from dust_extinction.baseclasses import BaseExtRvModel
from dust_extinction.shapes import _modified_drude, FM90


x_range_G22 = [1.0 / 35.0, 1.0 / 0.09]


class G22(BaseExtRvModel):
    r"""
    Gordon et al. (2022) Milky Way R(V) dependent model

    Parameters
    ----------
    Rv: float
        R(V) = A(V)/E(B-V) = total-to-selective extinction

    Raises
    ------
    InputParameterError
       Input Rv values outside of defined range

    Notes
    -----
    From Gordon et al. (2022, in prep.)

    Example showing CCM89 curves for a range of R(V) values.

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.parameter_averages import G22

        fig, ax = plt.subplots()

        # generate the curves and plot them
        x = np.arange(0.5,10.0,0.1)/u.micron

        Rvs = ['2.0','3.0','4.0','5.0','6.0']
        for cur_Rv in Rvs:
           ext_model = G22(Rv=cur_Rv)
           ax.plot(x,ext_model(x),label='R(V) = ' + str(cur_Rv))

        ax.set_xlabel(r'$x$ [$\mu m^{-1}$]')
        ax.set_ylabel(r'$A(x)/A(V)$')

        ax.legend(loc='best')
        plt.show()
    """

    Rv_range = [2.0, 6.0]
    x_range = x_range_G22

    def evaluate(self, in_x, Rv):
        """
        G22 function

        Parameters
        ----------
        in_x: float
           expects either x in units of wavelengths or frequency
           or assumes wavelengths in wavenumbers [1/micron]

           internally wavenumbers are used

        Returns
        -------
        axav: np array (float)
            A(x)/A(V) extinction curve [mag]

        Raises
        ------
        ValueError
           Input x values outside of defined range
        """
        x = _get_x_in_wavenumbers(in_x)

        # check that the wavenumbers are within the defined range
        _test_valid_x_range(x, self.x_range, "G22")

        # setup the a & b coefficient vectors
        n_x = len(x)
        self.a = np.zeros(n_x)
        self.b = np.zeros(n_x)

        # define the ranges
        ir_indxs = np.where(np.logical_and(1.0 / 35.0 <= x, x < 1.0 / 1.0))
        opt_indxs = np.where(np.logical_and(1.0 / 1.0 <= x, x < 1.0 / 0.3))
        uv_indxs = np.where(np.logical_and(1.0 / 0.3 <= x, x <= 1.0 / 0.09))

        # NIR/MIR

        # optical

        # Ultrviolet
        params_intercept = [
            0.84003414,
            0.28206022,
            1.05119322,
            0.11807519,
            4.59999236,
            0.99000999,
        ]
        fm90_model_a = FM90(
            C1=params_intercept[0],
            C2=params_intercept[1],
            C3=params_intercept[2],
            C4=params_intercept[3],
            xo=params_intercept[4],
            gamma=params_intercept[5],
        )
        self.a[uv_indxs] = fm90_model_a(x[uv_indxs])

        params_slope = [
            -3.47808977,
            2.15608243,
            4.35597383,
            1.01607364,
            4.60001101,
            0.98997783,
        ]
        fm90_model_b = FM90(
            C1=params_slope[0],
            C2=params_slope[1],
            C3=params_slope[2],
            C4=params_slope[3],
            xo=params_slope[4],
            gamma=params_slope[5],
        )
        self.b[uv_indxs] = fm90_model_b(x[uv_indxs])

        # return A(x)/A(V)
        return self.a + self.b / Rv


class G21mod(Fittable1DModel):
    r"""
    Gordon et al. (2021) powerlaw plus two modified Drude profiles
    (for the 10 & 20 micron silicate features)
    for the 1 to 40 micron A(lambda)/A(V) extinction curve.

    Parameters
    ----------
    scale: float
        amplitude of the powerlaw at 1 micron
    alpha: float
        power of powerlaw
    and more parameters

    Notes
    -----
    From Gordon et al. (2021, ApJ, submitted)

    Only applicable at NIR/MIR wavelengths from 1-40 micron

    Example showing a G21 curve with components identified.

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.shapes import G21

        fig, ax = plt.subplots()

        # generate the curves and plot them
        lam = np.logspace(np.log10(1.01), np.log10(39.9), num=1000)
        x = (1.0/lam)/u.micron

        ext_model = G21()
        ax.plot(1/x,ext_model(x),label='total')

        ext_model = G21(sil1_amp=0.0, sil2_amp=0.0)
        ax.plot(1./x,ext_model(x),label='power-law only')

        ext_model = G21(sil2_amp=0.0)
        ax.plot(1./x,ext_model(x),label='power-law+sil1 only')

        ext_model = G21(sil1_amp=0.0)
        ax.plot(1./x,ext_model(x),label='power-law+sil2 only')

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlabel('$\lambda$ [$\mu$m]')
        ax.set_ylabel('$A(x)/A(V)$')

        ax.set_title('G21')

        ax.legend(loc='best')
        plt.show()

    """

    # inputs = ("x",)
    # outputs = ("axav",)

    scale = Parameter(
        description="powerlaw: amplitude", default=0.37, bounds=(0.0, 1.0)
    )
    alpha = Parameter(description="powerlaw: alpha", default=1.7, bounds=(0.5, 5.0))
    alpha2 = Parameter(description="powerlaw: alpha2", default=1.4, bounds=(-0.5, 5.0))
    swave = Parameter(description="powerlaw: swave", default=4.0, bounds=(2.0, 10.0))
    ice_amp = Parameter(
        description="ice 3um: amplitude", default=0.0019, bounds=(0.0001, 0.3)
    )
    ice_center = Parameter(
        description="ice 3um: center", default=3.02, bounds=(2.9, 3.1)
    )
    ice_fwhm = Parameter(description="ice 3um: fwhm", default=0.45, bounds=(0.3, 0.6))
    ice_asym = Parameter(
        description="ice 3um: asymmetry", default=-1.0, bounds=(-2.0, 0.0)
    )
    sil1_amp = Parameter(
        description="silicate 10um: amplitude", default=0.07, bounds=(0.001, 0.3)
    )
    sil1_center = Parameter(
        description="silicate 10um: center", default=9.87, bounds=(8.0, 12.0)
    )
    sil1_fwhm = Parameter(
        description="silicate 10um: fwhm", default=2.5, bounds=(1.0, 10.0)
    )
    sil1_asym = Parameter(
        description="silicate 10um: asymmetry", default=-0.23, bounds=(-2.0, 2.0)
    )
    sil2_amp = Parameter(
        description="silicate 20um: amplitude", default=0.025, bounds=(0.001, 0.3)
    )
    sil2_center = Parameter(
        description="silicate 20um: center", default=17.0, bounds=(16.0, 24.0)
    )
    sil2_fwhm = Parameter(
        description="silicate 20um: fwhm", default=13.0, bounds=(5.0, 20.0)
    )
    sil2_asym = Parameter(
        description="silicate 20um: asymmetry", default=-0.27, bounds=(-2.0, 2.0)
    )

    x_range = [1.0 / 40.0, 1.0 / 0.8]

    def evaluate(
        self,
        in_x,
        scale,
        alpha,
        alpha2,
        swave,
        ice_amp,
        ice_center,
        ice_fwhm,
        ice_asym,
        sil1_amp,
        sil1_center,
        sil1_fwhm,
        sil1_asym,
        sil2_amp,
        sil2_center,
        sil2_fwhm,
        sil2_asym,
    ):
        """
        G21 function

        Parameters
        ----------
        in_x: float
           expects either x in units of wavelengths or frequency
           or assumes wavelengths in wavenumbers [1/micron]

        Returns
        -------
        axav: np array (float)
            A(x)/A(V) extinction curve [mag]

        Raises
        ------
        ValueError
           Input x values outside of defined range
        """
        x = _get_x_in_wavenumbers(in_x)

        # check that the wavenumbers are within the defined range
        _test_valid_x_range(x, self.x_range, "G21")

        wave = 1 / x

        # broken powerlaw
        # swave = 4.0
        axav = scale * (wave ** (-1.0 * alpha))
        (gindxs,) = np.where(wave > swave)
        if len(gindxs) > 0:
            norm_ratio = swave ** (-1.0 * alpha) / swave ** (-1.0 * alpha2)
            axav[gindxs] = scale * norm_ratio * (wave[gindxs] ** (-1.0 * alpha2))

        # silicate feature drudes
        axav += _modified_drude(wave, ice_amp, ice_center, ice_fwhm, ice_asym)
        axav += _modified_drude(wave, sil1_amp, sil1_center, sil1_fwhm, sil1_asym)
        axav += _modified_drude(wave, sil2_amp, sil2_center, sil2_fwhm, sil2_asym)

        return axav


class G22opt(Fittable1DModel):
    r"""
    Optical extinction curve model w/ drudes

    Parameters
    ----------
    scale: float
        amplitude of the powerlaw at 1 micron
    alpha: float
        power of powerlaw
    and more parameters
    """

    # inputs = ("x",)
    # outputs = ("axav",)

    scale = Parameter(
        description="powerlaw: amplitude", default=0.37, bounds=(0.0, 1.0)
    )
    alpha = Parameter(description="powerlaw: alpha", default=1.7, bounds=(0.5, 5.0))
    alpha2 = Parameter(description="powerlaw: alpha2", default=1.4, bounds=(-0.5, 5.0))
    swave = Parameter(description="powerlaw: swave", default=4.0, bounds=(2.0, 10.0))
    vss1_amp = Parameter(
        description="ice 3um: amplitude", default=0.0019, bounds=(0.0001, 0.3)
    )
    vss1_center = Parameter(
        description="ice 3um: center", default=3.02, bounds=(2.9, 3.1)
    )
    vss1_fwhm = Parameter(description="ice 3um: fwhm", default=0.45, bounds=(0.3, 0.6))
    vss1_asym = Parameter(
        description="ice 3um: asymmetry", default=-1.0, bounds=(-2.0, 0.0)
    )

    x_range = [1.0 / 1.0, 1.0 / 0.3]

    def evaluate(
        self,
        in_x,
        scale,
        alpha,
        alpha2,
        swave,
        vss1_amp,
        vss1_center,
        vss1_fwhm,
        vss1_asym,
    ):
        """
        G22opt function

        Parameters
        ----------
        in_x: float
           expects either x in units of wavelengths or frequency
           or assumes wavelengths in wavenumbers [1/micron]

        Returns
        -------
        axav: np array (float)
            A(x)/A(V) extinction curve [mag]

        Raises
        ------
        ValueError
           Input x values outside of defined range
        """
        x = _get_x_in_wavenumbers(in_x)

        # check that the wavenumbers are within the defined range
        _test_valid_x_range(x, self.x_range, "G22opt")

        wave = 1 / x

        # broken powerlaw
        # swave = 4.0
        axav = scale * (wave ** (-1.0 * alpha))
        (gindxs,) = np.where(wave > swave)
        if len(gindxs) > 0:
            norm_ratio = swave ** (-1.0 * alpha) / swave ** (-1.0 * alpha2)
            axav[gindxs] = scale * norm_ratio * (wave[gindxs] ** (-1.0 * alpha2))

        # silicate feature drudes
        axav += _modified_drude(wave, vss1_amp, vss1_center, vss1_fwhm, vss1_asym)

        return axav


class G22pow(Fittable1DModel):
    r"""
    Double powerlaw fit versus inverse microns

    Parameters
    ----------
    scale: float
        amplitude of the powerlaw at 1 micron
    alpha: float
        power of powerlaw
    """

    # inputs = ("x",)
    # outputs = ("axav",)

    scale = Parameter(
        description="powerlaw: amplitude", default=-0.8, bounds=(-2.0, 0.0)
    )
    alpha = Parameter(description="powerlaw: alpha", default=1.7, bounds=(0.5, 5.0))
    alpha2 = Parameter(description="powerlaw: alpha2", default=1.4, bounds=(-0.5, 5.0))
    swave = Parameter(description="powerlaw: swave", default=4.0, bounds=(2.0, 10.0))

    x_range = [1.0 / 40.0, 1.0 / 0.8]

    def evaluate(
        self,
        in_x,
        scale,
        alpha,
        alpha2,
        swave,
    ):
        """
        G21 function

        Parameters
        ----------
        in_x: float
           expects either x in units of wavelengths or frequency
           or assumes wavelengths in wavenumbers [1/micron]

        Returns
        -------
        axav: np array (float)
            A(x)/A(V) extinction curve [mag]

        Raises
        ------
        ValueError
           Input x values outside of defined range
        """
        x = _get_x_in_wavenumbers(in_x)

        # check that the wavenumbers are within the defined range
        _test_valid_x_range(x, self.x_range, "G21")

        wave = 1 / x

        # broken powerlaw
        # swave = 4.0
        axav = scale * (wave ** (-1.0 * alpha))
        (gindxs,) = np.where(wave > swave)
        if len(gindxs) > 0:
            norm_ratio = swave ** (-1.0 * alpha) / swave ** (-1.0 * alpha2)
            axav[gindxs] = scale * norm_ratio * (wave[gindxs] ** (-1.0 * alpha2))

        return axav
