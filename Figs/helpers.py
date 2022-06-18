import numpy as np
from scipy.special import comb
from numpy.random import default_rng
import astropy.units as u

from astropy.modeling import Fittable1DModel, Parameter
from astropy.modeling.models import Drude1D, Polynomial1D, PowerLaw1D

from astropy.stats import sigma_clip
from astropy.modeling.models import Linear1D
from astropy.modeling.fitting import (
    LinearLSQFitter,
    FittingWithOutlierRemoval,
    # LevMarLSQFitter,
)

# from dust_extinction.shapes import G21
from dust_extinction.helpers import _get_x_in_wavenumbers, _test_valid_x_range
from dust_extinction.baseclasses import BaseExtRvModel
from dust_extinction.shapes import _modified_drude, FM90


x_range_G22 = [1.0 / 45.0, 1.0 / 0.08]


def smoothstep(x, x_min=0, x_max=1, N=1):
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)

    result = 0
    for n in range(0, N + 1):
        result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n

    result *= x ** (N + 1)

    return result


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

        # overlap ranges
        optir_waves = [1.0, 1.1]
        optir_overlap = (x >= 1.0 / optir_waves[1]) & (x <= 1.0 / optir_waves[0])
        uvopt_waves = [0.3, 0.33]
        uvopt_overlap = (x >= 1.0 / uvopt_waves[1]) & (x <= 1.0 / uvopt_waves[0])

        # NIR/MIR
        # fmt: off
        # (scale, alpha1, alpha2, swave, swidth), sil1, sil2
        ir_a = [0.38982, 1.72802, 1.02603, 3.92291, 8.37168,
                0.06434, 9.81785, 2.17175, -0.28987,
                0.02826, 19.78533, 17., -0.27]
        # ir_b = [-1.07917, 1., -1.18025]
        ir_b = [-1.02151, 1., -1.17596]
        # ir_b = [-0.3941, 12.18675, -101.28211, 333.40123, -536.15058,
        #         415.3248, -123.82645]
        # fmt: on
        g21mod = G21mod()
        g21mod.parameters = ir_a
        self.a[ir_indxs] = g21mod(x[ir_indxs] / u.micron)

        irpow = PowerLaw1D()
        irpow.parameters = ir_b
        self.b[ir_indxs] = irpow(x[ir_indxs])
        # irpoly = Polynomial1D(6)
        # irpoly.parameters = ir_b
        # self.b[ir_indxs] = irpoly(x[ir_indxs])

        # optical
        # fmt: off
        # polynomial coeffs, ISS1, ISS2, ISS3
        opt_a = [-0.70783, 1.42617, -0.43445, 0.10785, -0.01129,
                 0.04326, 2.288, 0.243,
                 0.03416, 2.054, 0.179,
                 0.01197, 1.587, 0.243]
        opt_b = [-1.12257, 0.50781, -0.72101, 0.55661, -0.08087,
                 0.22524, 2.288, 0.243,
                 0.30337, 2.054, 0.179,
                 0.11669, 1.587, 0.243]
        # fmt: on
        m20_model_a = Polynomial1D(4) + Drude1D() + Drude1D() + Drude1D()
        m20_model_a.parameters = opt_a
        self.a[opt_indxs] = m20_model_a(x[opt_indxs])
        m20_model_b = Polynomial1D(4) + Drude1D() + Drude1D() + Drude1D()
        m20_model_b.parameters = opt_b
        self.b[opt_indxs] = m20_model_b(x[opt_indxs])

        # overlap between optical/ir
        weights = (1.0 / optir_waves[1] - x[optir_overlap]) / (
            1.0 / optir_waves[1] - 1.0 / optir_waves[0]
        )
        self.a[optir_overlap] = weights * m20_model_a(x[optir_overlap])
        self.a[optir_overlap] += (1.0 - weights) * g21mod(x[optir_overlap] / u.micron)
        self.b[optir_overlap] = weights * m20_model_b(x[optir_overlap])
        self.b[optir_overlap] += (1.0 - weights) * irpow(x[optir_overlap])

        # Ultraviolet
        uv_a = [0.81796, 0.27968, 1.02785, 0.11001, 4.59999, 0.99004]
        uv_b = [-2.98641, 1.89734, 3.3836, 0.63903, 4.60001, 0.99007]

        fm90_model_a = FM90()
        fm90_model_a.parameters = uv_a
        self.a[uv_indxs] = fm90_model_a(x[uv_indxs] / u.micron)
        fm90_model_b = FM90()
        fm90_model_b.parameters = uv_b
        self.b[uv_indxs] = fm90_model_b(x[uv_indxs] / u.micron)

        # overlap between uv/optical
        weights = (1.0 / uvopt_waves[1] - x[uvopt_overlap]) / (
            1.0 / uvopt_waves[1] - 1.0 / uvopt_waves[0]
        )
        self.a[uvopt_overlap] = weights * fm90_model_a(x[uvopt_overlap] / u.micron)
        self.a[uvopt_overlap] += (1.0 - weights) * m20_model_a(x[uvopt_overlap])
        self.b[uvopt_overlap] = weights * fm90_model_b(x[uvopt_overlap] / u.micron)
        self.b[uvopt_overlap] += (1.0 - weights) * m20_model_b(x[uvopt_overlap])

        # return A(x)/A(V)
        return self.a + self.b * (1 / Rv - 1 / 3.1)


class G22LFnoweight(BaseExtRvModel):
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

        # overlap ranges
        optir_waves = [1.0, 1.1]
        optir_overlap = (x >= 1.0 / optir_waves[1]) & (x <= 1.0 / optir_waves[0])
        uvopt_waves = [0.3, 0.33]
        uvopt_overlap = (x >= 1.0 / uvopt_waves[1]) & (x <= 1.0 / uvopt_waves[0])

        # NIR/MIR
        # fmt: off
        # (scale, alpha1, alpha2, swave, swidth), sil1, sil2
        ir_a = [0.37811, 1.61943, 0.67069, 4.19323, 6.58017,
                0.06214, 9.82638, 2.12157, -0.26485,
                0.02436, 20.22223, 17., -0.27]
        # ir_b = [-1.07917, 1., -1.18025]
        ir_b = [-0.87888, 1., -1.12937]
        ir_b = [-0.3941, 12.18675, -101.28211, 333.40123, -536.15058,
                415.3248, -123.82645]
        # fmt: on
        g21mod = G21mod()
        g21mod.parameters = ir_a
        self.a[ir_indxs] = g21mod(x[ir_indxs] / u.micron)

        # irpow = PowerLaw1D()
        irpoly = Polynomial1D(6)
        irpoly.parameters = ir_b
        self.b[ir_indxs] = irpoly(x[ir_indxs])

        # optical
        # fmt: off
        # polynomial coeffs, ISS1, ISS2, ISS3
        opt_a = [-0.88859, 1.84765, -0.76068, 0.21211, -0.02319,
                 0.04612, 2.288, 0.243,
                 0.03272, 2.054, 0.179,
                 0.01283, 1.587, 0.243]
        opt_b = [-0.51533, -0.60254, 0.26787, 0.17602, -0.03303,
                 0.31056, 2.288, 0.243,
                 0.15588, 2.054, 0.179,
                 0.1077, 1.587, 0.243]
        # fmt: on
        m20_model_a = Polynomial1D(4) + Drude1D() + Drude1D() + Drude1D()
        m20_model_a.parameters = opt_a
        self.a[opt_indxs] = m20_model_a(x[opt_indxs])
        m20_model_b = Polynomial1D(4) + Drude1D() + Drude1D() + Drude1D()
        m20_model_b.parameters = opt_b
        self.b[opt_indxs] = m20_model_b(x[opt_indxs])

        # overlap between optica/ir
        weights = (1.0 / optir_waves[1] - x[optir_overlap]) / (
            1.0 / optir_waves[1] - 1.0 / optir_waves[0]
        )
        self.a[optir_overlap] = weights * m20_model_a(x[optir_overlap])
        self.a[optir_overlap] += (1.0 - weights) * g21mod(x[optir_overlap] / u.micron)
        self.b[optir_overlap] = weights * m20_model_b(x[optir_overlap])
        self.b[optir_overlap] += (1.0 - weights) * irpoly(x[optir_overlap])

        # Ultraviolet
        uv_a = [0.77653, 0.27609, 1.12978, 0.10893, 4.59999, 0.99002]
        uv_b = [-2.54777, 1.51309, 3.79311, 0.40683, 4.6, 0.99001]

        fm90_model_a = FM90()
        fm90_model_a.parameters = uv_a
        self.a[uv_indxs] = fm90_model_a(x[uv_indxs] / u.micron)
        fm90_model_b = FM90()
        fm90_model_b.parameters = uv_b
        self.b[uv_indxs] = fm90_model_b(x[uv_indxs] / u.micron)

        # overlap between uv/optical
        weights = (1.0 / uvopt_waves[1] - x[uvopt_overlap]) / (
            1.0 / uvopt_waves[1] - 1.0 / uvopt_waves[0]
        )
        self.a[uvopt_overlap] = weights * fm90_model_a(x[uvopt_overlap] / u.micron)
        self.a[uvopt_overlap] += (1.0 - weights) * m20_model_a(x[uvopt_overlap])
        self.b[uvopt_overlap] = weights * fm90_model_b(x[uvopt_overlap] / u.micron)
        self.b[uvopt_overlap] += (1.0 - weights) * m20_model_b(x[uvopt_overlap])

        # return A(x)/A(V)
        return self.a + self.b * (1 / Rv - 1 / 3.1)


class G22MC(BaseExtRvModel):
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

        # overlap ranges
        optir_waves = [1.0, 1.1]
        optir_overlap = (x >= 1.0 / optir_waves[1]) & (x <= 1.0 / optir_waves[0])
        uvopt_waves = [0.3, 0.33]
        uvopt_overlap = (x >= 1.0 / uvopt_waves[1]) & (x <= 1.0 / uvopt_waves[0])

        # NIR/MIR
        # fmt: off
        # (scale, alpha1, alpha2, swave, swidth), sil1, sil2
        ir_a = [0.3798, 1.62151, 0.69375, 4.18005, 6.60928,
                0.06182, 9.82574, 2.131, -0.26983,
                0.02438, 20.13538, 17., -0.27]
        # ir_b = [-1.07917, 1., -1.18025]
        ir_b = [-0.36212, 11.23189, -93.7538, 309.70542, -498.89145,
                386.78939, -115.36082]
        # fmt: on
        g21mod = G21mod()
        g21mod.parameters = ir_a
        self.a[ir_indxs] = g21mod(x[ir_indxs] / u.micron)

        # irpow = PowerLaw1D()
        irpoly = Polynomial1D(6)
        irpoly.parameters = ir_b
        self.b[ir_indxs] = irpoly(x[ir_indxs])

        # optical
        # fmt: off
        # polynomial coeffs, ISS1, ISS2, ISS3
        opt_a = [-0.86754, 1.80933, -0.73544, 0.20491, -0.02244,
                 0.04622, 2.288, 0.243,
                 0.03303, 2.054, 0.179,
                 0.01251, 1.587, 0.243]
        opt_b = [0.18594, -1.71886, 0.92739, 0.00489, -0.01672,
                 0.31003, 2.288, 0.243,
                 0.15997, 2.054, 0.179,
                 0.09936, 1.587, 0.243]
        # fmt: on
        m20_model_a = Polynomial1D(4) + Drude1D() + Drude1D() + Drude1D()
        m20_model_a.parameters = opt_a
        self.a[opt_indxs] = m20_model_a(x[opt_indxs])
        m20_model_b = Polynomial1D(4) + Drude1D() + Drude1D() + Drude1D()
        m20_model_b.parameters = opt_b
        self.b[opt_indxs] = m20_model_b(x[opt_indxs])

        # overlap between optica/ir
        weights = (1.0 / optir_waves[1] - x[optir_overlap]) / (
            1.0 / optir_waves[1] - 1.0 / optir_waves[0]
        )
        self.a[optir_overlap] = weights * m20_model_a(x[optir_overlap])
        self.a[optir_overlap] += (1.0 - weights) * g21mod(x[optir_overlap] / u.micron)
        self.b[optir_overlap] = weights * m20_model_b(x[optir_overlap])
        self.b[optir_overlap] += (1.0 - weights) * irpoly(x[optir_overlap])

        # Ultraviolet
        uv_a = [0.79959, 0.26926, 1.10624, 0.10581, 4.59999, 0.99003]
        uv_b = [-1.22067, 1.12291, 2.32273, 0.15026, 4.60002, 0.98999]

        fm90_model_a = FM90()
        fm90_model_a.parameters = uv_a
        self.a[uv_indxs] = fm90_model_a(x[uv_indxs] / u.micron)
        fm90_model_b = FM90()
        fm90_model_b.parameters = uv_b
        self.b[uv_indxs] = fm90_model_b(x[uv_indxs] / u.micron)

        # overlap between uv/optical
        weights = (1.0 / uvopt_waves[1] - x[uvopt_overlap]) / (
            1.0 / uvopt_waves[1] - 1.0 / uvopt_waves[0]
        )
        self.a[uvopt_overlap] = weights * fm90_model_a(x[uvopt_overlap] / u.micron)
        self.a[uvopt_overlap] += (1.0 - weights) * m20_model_a(x[uvopt_overlap])
        self.b[uvopt_overlap] = weights * fm90_model_b(x[uvopt_overlap] / u.micron)
        self.b[uvopt_overlap] += (1.0 - weights) * m20_model_b(x[uvopt_overlap])

        # return A(x)/A(V)
        return self.a + self.b * (1 / Rv - 1 / 3.1)


class G22HF(BaseExtRvModel):
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

        # overlap ranges
        optir_waves = [1.0, 1.1]
        optir_overlap = (x >= 1.0 / optir_waves[1]) & (x <= 1.0 / optir_waves[0])
        uvopt_waves = [0.3, 0.33]
        uvopt_overlap = (x >= 1.0 / uvopt_waves[1]) & (x <= 1.0 / uvopt_waves[0])

        # NIR/MIR
        # fmt: off
        # (scale, alpha1, alpha2, swave, swidth), sil1, sil2
        ir_a = [0.37964, 1.60228, 0.44087, 4.43503, 6.70447,
                0.06022, 9.82174, 1.91943, -0.28325,
                0.02695, 20.38638, 17., -0.27]
        # ir_b = [-1.07917, 1., -1.18025]
        ir_b = [-0.54372, 16.03173, -128.15723, 414.0907, -659.84316,
                510.29657, -152.88959]
        # fmt: on
        g21mod = G21mod()
        g21mod.parameters = ir_a
        self.a[ir_indxs] = g21mod(x[ir_indxs] / u.micron)

        # irpow = PowerLaw1D()
        irpoly = Polynomial1D(6)
        irpoly.parameters = ir_b
        self.b[ir_indxs] = irpoly(x[ir_indxs])

        # optical
        # fmt: off
        # polynomial coeffs, ISS1, ISS2, ISS3
        opt_a = [-0.8792, 1.82004, -0.74087, 0.20642, -0.02249,
                 0.04365, 2.288, 0.243,
                 0.03358, 2.054, 0.179,
                 0.01563, 1.587, 0.243]
        opt_b = [-0.62879, -1.30082, 1.08796, -0.13022, 0.00883,
                 0.19705, 2.288, 0.243,
                 0.18687, 2.054, 0.179,
                 0.25191, 1.587, 0.243]
        # fmt: on
        m20_model_a = Polynomial1D(4) + Drude1D() + Drude1D() + Drude1D()
        m20_model_a.parameters = opt_a
        self.a[opt_indxs] = m20_model_a(x[opt_indxs])
        m20_model_b = Polynomial1D(4) + Drude1D() + Drude1D() + Drude1D()
        m20_model_b.parameters = opt_b
        self.b[opt_indxs] = m20_model_b(x[opt_indxs])

        # overlap between optica/ir
        weights = (1.0 / optir_waves[1] - x[optir_overlap]) / (
            1.0 / optir_waves[1] - 1.0 / optir_waves[0]
        )
        self.a[optir_overlap] = weights * m20_model_a(x[optir_overlap])
        self.a[optir_overlap] += (1.0 - weights) * g21mod(x[optir_overlap] / u.micron)
        self.b[optir_overlap] = weights * m20_model_b(x[optir_overlap])
        self.b[optir_overlap] += (1.0 - weights) * irpoly(x[optir_overlap])

        # Ultraviolet
        uv_a = [0.7781, 0.28296, 1.12103, 0.12758, 4.59999, 0.99004]
        uv_b = [-4.4206, 2.19942, 5.19608, 0.97013, 4.6, 0.99001]

        fm90_model_a = FM90()
        fm90_model_a.parameters = uv_a
        self.a[uv_indxs] = fm90_model_a(x[uv_indxs] / u.micron)
        fm90_model_b = FM90()
        fm90_model_b.parameters = uv_b
        self.b[uv_indxs] = fm90_model_b(x[uv_indxs] / u.micron)

        # overlap between uv/optical
        weights = (1.0 / uvopt_waves[1] - x[uvopt_overlap]) / (
            1.0 / uvopt_waves[1] - 1.0 / uvopt_waves[0]
        )
        self.a[uvopt_overlap] = weights * fm90_model_a(x[uvopt_overlap] / u.micron)
        self.a[uvopt_overlap] += (1.0 - weights) * m20_model_a(x[uvopt_overlap])
        self.b[uvopt_overlap] = weights * fm90_model_b(x[uvopt_overlap] / u.micron)
        self.b[uvopt_overlap] += (1.0 - weights) * m20_model_b(x[uvopt_overlap])

        # return A(x)/A(V)
        return self.a + self.b * (1 / Rv - 1 / 3.1)


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
    swidth = Parameter(description="powerlaw: swidth", default=1.0, bounds=(0.5, 1.5))
    # ice_amp = Parameter(
    #     description="ice 3um: amplitude", default=0.0019, bounds=(0.0001, 0.3)
    # )
    # ice_center = Parameter(
    #     description="ice 3um: center", default=3.02, bounds=(2.9, 3.1)
    # )
    # ice_fwhm = Parameter(description="ice 3um: fwhm", default=0.45, bounds=(0.3, 0.6))
    # ice_asym = Parameter(
    #     description="ice 3um: asymmetry", default=-1.0, bounds=(-2.0, 0.0)
    # )
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
        swidth,
        # ice_amp,
        # ice_center,
        # ice_fwhm,
        # ice_asym,
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

        # broken powerlaw with a smooth transition
        axav_pow1 = scale * (wave ** (-1.0 * alpha))

        norm_ratio = swave ** (-1.0 * alpha) / swave ** (-1.0 * alpha2)
        axav_pow2 = scale * norm_ratio * (wave ** (-1.0 * alpha2))

        # use smoothstep to smoothly transition between the two powerlaws
        weights = smoothstep(
            wave, x_min=swave - swidth / 2, x_max=swave + swidth / 2, N=1
        )
        # weights = (wave - (swave - swidth / 2)) / swidth
        # weights[wave < (swave - swidth / 2)] = 0.0
        # weights[wave > (swave + swidth / 2)] = 0.0
        axav = axav_pow1 * (1.0 - weights) + axav_pow2 * weights

        # import matplotlib.pyplot as plt
        #
        # plt.plot(wave, weights)
        # plt.show()
        # exit()

        # silicate feature drudes
        # axav += _modified_drude(wave, ice_amp, ice_center, ice_fwhm, ice_asym)
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


def mcfit_cov(x, y, covs, mask, num=10, ax=None):
    """
    Do a number of linear fits using Monte Carlo to resample the data based on
    the covariance matrices.

    Returns
    -------
    fitparams : 2d ndarray
        2 x num array with intercept and slope fit parameters for num fits
    """
    fit = LinearLSQFitter()
    line_init = Linear1D()
    or_fit = FittingWithOutlierRemoval(fit, sigma_clip, niter=3, sigma=3.0)

    rng = default_rng(seed=12345)

    # generate a new set of x,y values based on the covariances
    npts = np.sum(mask)
    newxs = np.zeros((npts, num))
    newys = np.zeros((npts, num))
    k2 = 0
    for k, (x1, y1, cov1) in enumerate(zip(x, y, covs)):
        if mask[k]:
            newxy = rng.multivariate_normal([x1, y1], cov1, size=num)
            newxs[k2, :] = newxy[:, 0]
            newys[k2, :] = newxy[:, 1]
            # newxs[k2, :] = x1
            # newys[k2, :] = y1
            k2 += 1

    # x = np.average(newxs, axis=1)
    # y = np.average(newys, axis=1)
    # fitted_line, mask = or_fit(line_init, x, y)
    # if ax is not None:
    #     ax.plot(x, fitted_line(x), "g:", alpha=0.75, lw=4)

    # print(y - np.average(newys, axis=1))
    # exit()

    fitparam = np.zeros((num, 2))
    # newx = np.zeros((npts))
    for k in range(num):
        # fitted_line = fit(line_init, newxs[:, k], newys[:, k])
        # fitted_line = fit(line_init, x, y)
        fitted_line, mask = or_fit(line_init, newxs[:, k], newys[:, k])
        fitparam[k, 0] = fitted_line.intercept.value
        fitparam[k, 1] = fitted_line.slope.value

        # if ax is not None:
        #    ax.plot(newxs[:, k], newys[:, k], "rx", alpha=0.15)

    return fitparam


def mcfit_cov_quad(x, y, covs, mask, num=10, ax=None):
    """
    Do a number of quadratic fits using Monte Carlo to resample the data based on
    the covariance matrices.

    Returns
    -------
    fitparams : 2d ndarray
        2 x num array with intercept and slope fit parameters for num fits
    """
    # fit = LevMarLSQFitter()
    fit = LinearLSQFitter()

    quad_init = Polynomial1D(2)
    or_fit = FittingWithOutlierRemoval(fit, sigma_clip, niter=3, sigma=3.0)

    rng = default_rng(seed=12345)

    # generate a new set of x,y values based on the covariances
    npts = np.sum(mask)
    newxs = np.zeros((npts, num))
    newys = np.zeros((npts, num))
    k2 = 0
    for k, (x1, y1, cov1) in enumerate(zip(x, y, covs)):
        if mask[k]:
            newxy = rng.multivariate_normal([x1, y1], cov1, size=num)
            newxs[k2, :] = newxy[:, 0]
            newys[k2, :] = newxy[:, 1]
            k2 += 1

    # x = np.average(newxs, axis=1)
    # y = np.average(newys, axis=1)
    # fitted_line, mask = or_fit(line_init, x, y)
    # if ax is not None:
    #     ax.plot(x, fitted_line(x), "g:", alpha=0.75, lw=4)

    # print(y - np.average(newys, axis=1))
    # exit()

    fitparam = np.zeros((num, 3))
    # newx = np.zeros((npts))
    for k in range(num):
        # fitted_line = fit(line_init, newxs[:, k], newys[:, k])
        # fitted_line = fit(line_init, x, y)
        fitted_line, mask = or_fit(quad_init, newxs[:, k], newys[:, k])
        fitparam[k, 0] = fitted_line.c0.value
        fitparam[k, 1] = fitted_line.c1.value
        fitparam[k, 2] = fitted_line.c2.value

        # if ax is not None:
        #     ax.plot(newxs[:, k], newys[:, k], "rx", alpha=0.15)

    return fitparam
