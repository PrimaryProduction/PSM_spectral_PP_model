r"""Clear-sky spectral solar irradiance at the sea surface.

A simple spectral model of direct and diffuse irradiance under cloudless
skies, after Bird (1984) and Bird & Riordan (1986), as adapted for ocean
primary production work by Sathyendranath & Platt (1988) and Gregg & Carder
(1990). The extraterrestrial spectrum is attenuated by

* Rayleigh (molecular) scattering,
* aerosol scattering and absorption (Angstrom law),
* water-vapour absorption,
* ozone absorption, and
* absorption by uniformly mixed gases (O2, CO2),

and the scattered light is redistributed into a diffuse sky component that
includes multiple reflections between the sea and the sky. The irradiances
returned are in W m-2 um-1; :func:`surface_light.instant` converts them to
quanta and applies cloud and surface corrections.

Attributes
----------
cfdict : dict
    Spectral coefficients (see :func:`coefficients`) evaluated on the
    default grid :data:`config.wavelengths`. Used by :class:`Irradiance`
    when no wavelengths are given.

References
----------
.. [1] Bird, R.E. (1984). A simple, solar spectral model for direct-normal
   and diffuse horizontal irradiance. *Solar Energy*, 32(4), 461-471.
   https://doi.org/10.1016/0038-092X(84)90260-3
.. [2] Bird, R.E. & Riordan, C. (1986). Simple solar spectral model for
   direct and diffuse irradiance on horizontal and tilted planes at the
   Earth's surface for cloudless atmospheres. *Journal of Climate and Applied
   Meteorology*, 25(1), 87-97.
   https://doi.org/10.1175/1520-0450(1986)025%3C0087:SSSMFD%3E2.0.CO;2
.. [3] Leckner, B. (1978). The spectral distribution of solar radiation at
   the Earth's surface - elements of a model. *Solar Energy*, 20(2), 143-150.
   https://doi.org/10.1016/0038-092X(78)90187-1
.. [4] Kasten, F. (1966). A new table and approximation formula for the
   relative optical air mass. *Archiv fur Meteorologie, Geophysik und
   Bioklimatologie, Serie B*, 14, 206-223. https://doi.org/10.1007/BF02248840
.. [5] Paltridge, G.W. & Platt, C.M.R. (1976). *Radiative Processes in
   Meteorology and Climatology*. Elsevier, Amsterdam.
.. [6] Sathyendranath, S. & Platt, T. (1988). The spectral irradiance field
   at the surface and in the interior of the ocean: a model for applications
   in oceanography and remote sensing. *Journal of Geophysical Research*,
   93(C8), 9270-9280. https://doi.org/10.1029/JC093iC08p09270
.. [7] Gregg, W.W. & Carder, K.L. (1990). A simple spectral solar irradiance
   model for cloudless maritime atmospheres. *Limnology and Oceanography*,
   35(8), 1657-1675. https://doi.org/10.4319/lo.1990.35.8.1657
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import KDTree

import config

def interpolate_lut(wavelengths):
    """Interpolate the diffuse-irradiance correction table to new wavelengths.

    Parameters
    ----------
    wavelengths : array_like
        Wavelengths (nm), within 400-710 nm.

    Returns
    -------
    numpy.ndarray
        Correction factors, shape ``(n_wavelengths, 7)``. Column ``j``
        applies to the solar zenith angle ``Irradiance.cd[j]``
        (0, 37, 48.19, 60, 70, 75 and 80 degrees).

    Raises
    ------
    ValueError
        If any wavelength is outside 400-710 nm.

    Notes
    -----
    The factors are the empirical wavelength- and zenith-dependent
    corrections that Bird (1984) derived to bring the simple diffuse
    irradiance model into agreement with a rigorous radiative-transfer code.
    They are tabulated at 400, 450, 500, 550 and 710 nm and linearly
    interpolated in wavelength.

    References
    ----------
    .. [1] Bird, R.E. (1984), https://doi.org/10.1016/0038-092X(84)90260-3
    """
    c = np.asarray(
        [
            [1.11, 1.13, 1.18, 1.24, 1.46, 1.70, 2.61],
            [1.04, 1.05, 1.09, 1.11, 1.24, 1.34, 1.72],
            [1.15, 1.00, 1.00, 0.99, 1.06, 1.07, 1.22],
            [1.12, 0.96, 0.96, 0.94, 0.99, 0.96, 1.04],
            [1.32, 1.12, 1.07, 1.02, 1.10, 0.90, 0.80]
        ]
    )
    lut_wl = np.array([400,450,500,550,710])
    intr = interp1d(lut_wl, c, axis=0)
    return intr(wavelengths)

def coefficients(wavelengths=None):
    """Spectral coefficients of the atmospheric model on a wavelength grid.

    Parameters
    ----------
    wavelengths : array_like, optional
        Wavelengths (nm). Defaults to :data:`config.wavelengths`. The
        tabulated values (24 bands between 400 and 710 nm) are linearly
        interpolated.

    Returns
    -------
    dict of numpy.ndarray
        ``"ao"``
            Ozone absorption coefficient ((atm cm)-1).
        ``"av"``
            Water-vapour absorption coefficient (cm-1).
        ``"ozone_weights"``
            Weights (0-1) selecting the bands where the transmittance of
            uniformly mixed gases is applied to the direct beam and the sky
            albedo (non-zero only towards 710 nm, the O2 band region).
        ``"dif_weights"``
            Weights (0-1) selecting the bands where the transmittance of
            uniformly mixed gases is applied to the diffuse irradiance
            (non-zero around 690 nm, the O2 B-band).
        ``"extspir"``
            Extraterrestrial spectral irradiance at mean Earth-Sun distance
            (W m-2 um-1).
        ``"wavelengths"``
            The wavelength grid (nm).
        ``"correction_lut"``
            Diffuse correction factors from :func:`interpolate_lut`,
            shape ``(n_wavelengths, 7)``.

    Notes
    -----
    Coefficients are those of the Bird (1984) / Bird & Riordan (1986) model
    at the 24 wavelengths used in the original Fortran implementation of the
    PSM. Interpolating the 0/1 weight arrays to a finer grid gives
    fractional weights in the neighbouring bands.
    """
    wavelengths = config.wavelengths if wavelengths is None else wavelengths
    # wavelengths at which to calculate transmittance)
    wls = np.asarray(
            [400., 410., 420., 430., 440., 450., 460., 470.,
             480., 490., 500., 510., 520., 530., 540., 550.,
             570., 593., 610., 630., 656., 667.6, 690., 710.])
    origdict = dict(
        # ozone absorption coefficient
        ao = np.asarray([0.,    0.,    0.,    0.,    0.,    0.003, 0.006, 0.009,
                         0.014, 0.021, 0.030, 0.040, 0.048, 0.063, 0.075, 0.095,
                         0.120, 0.119, 0.132, 0.120, 0.065, 0.060, 0.028, 0.018]),
        # water vapour absorption coefficient
        av = np.asarray([0., 0., 0., 0., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0., 0., 0., 0.,
                         0., 0.075, 0., 0., 0., 0., 0.016, 0.0125]),
        ozone_weights = np.asarray([0., 0., 0., 0., 0., 0., 0., 0.,
                                    0., 0., 0., 0., 0., 0., 0., 0.,
                                    0., 0, 0., 0., 0., 0., 0., 1]),
        dif_weights = np.asarray([0., 0., 0., 0., 0., 0., 0., 0.,
                                    0., 0., 0., 0., 0., 0., 0., 0.,
                                    0., 0, 0., 0., 0., 0., 1., 0]),
        # Extra terrestrial spectral irradiance
        extspir = np.asarray(
            [1479.1, 1701.3, 1740.4, 1587.2, 1837.0, 2005.0, 2043.0, 1987.0,
             2027.0, 1896.0, 1909.0, 1927.0, 1831.0, 1891.0, 1898.0, 1892.0,
             1840.0, 1768.0, 1728.0, 1658.0, 1524.0, 1531.0, 1420.0, 1399.0]),
        )

    outdict =  {key:np.interp(wavelengths, wls, val) for key,val in origdict.items()}
    outdict["wavelengths"] = wavelengths
    outdict["correction_lut"] = interpolate_lut(wavelengths)
    return outdict

cfdict = coefficients()




class Irradiance:
    r"""Clear-sky direct and diffuse spectral irradiance for given zenith angles.

    All transmittances and irradiances are exposed as lazily evaluated
    properties, each with a short alias matching the variable names of the
    original Fortran code (for example :attr:`tr` for
    :attr:`rayleigh_transmittance`).

    Parameters
    ----------
    zenith_deg : float or array_like
        Solar zenith angle(s) (degrees). Arrays of any shape are supported;
        the wavelength axis is appended as the last dimension of the outputs.
    wavelengths : array_like, optional
        Wavelengths (nm), within 400-710 nm. Defaults to
        :data:`config.wavelengths`.

    Attributes
    ----------
    alpha1, beta1 : float
        Angstrom exponent and turbidity coefficient of the aerosol optical
        thickness below 500 nm (Bird & Riordan 1986).
    alpha2, beta2 : float
        As ``alpha1``/``beta1`` for 500 nm and longer.
    cd : numpy.ndarray
        Zenith angles (degrees) of the columns of the diffuse correction
        table (see :func:`interpolate_lut`).
    zenith_deg, zenith_rad : float or numpy.ndarray
        Solar zenith angle in degrees and radians.
    cfdict : dict
        Spectral coefficients from :func:`coefficients`.
    wld : numpy.ndarray
        Wavelengths in micrometres.

    Notes
    -----
    The sky albedo :attr:`air_albedo` is evaluated once at construction time
    with a fixed air mass of 1.9 (following Bird & Riordan's use of a fixed
    air mass for the sky reflectivity) and frozen; afterwards the air mass is
    computed from the zenith angle.

    References
    ----------
    .. [1] Bird, R.E. (1984), https://doi.org/10.1016/0038-092X(84)90260-3
    .. [2] Bird, R.E. & Riordan, C. (1986),
       https://doi.org/10.1175/1520-0450(1986)025%3C0087:SSSMFD%3E2.0.CO;2

    Examples
    --------
    >>> import numpy as np
    >>> from irradiance import Irradiance
    >>> ir = Irradiance(np.array([10.0, 60.0]))
    >>> ir.direct_irradiance.shape, ir.diffuse_irradiance.shape
    ((2, 61), (2, 61))
    """

    alpha1 = 1.0274
    beta1  = 0.1324
    alpha2 = 1.206
    beta2  = 0.117
    cd = np.asarray([0., 37., 48.19, 60., 70., 75., 80.])
    _airmass = None
    _air_albedo = None

    def __init__(self, zenith_deg, wavelengths=None):
        self.zenith_deg = zenith_deg #np.minimum(zenith_deg, 79.5)
        self.zenith_rad = np.deg2rad(self.zenith_deg)
        if wavelengths is None:
            self.cfdict = cfdict
        else:
            self.cfdict = coefficients(wavelengths)
        self.wld = self.cfdict["wavelengths"] / 1000.

        # Calculate air_albedo with a specified airmass
        self.airmass = 1.9 # Set airmas to a specific number
        self.air_albedo = self.air_albedo # Fix air_albedo
        self.airmass = None # Reset airmass to be calculated


    @property
    def airmass(self):
        r"""Relative optical air mass :math:`M` (dimensionless).

        Kasten's (1966) approximation, bounded below by 1:

        $$
        M = \left[\cos\theta_s + 0.15\,(93.885 - \theta_s)^{-1.253}\right]^{-1},
        $$

        with :math:`\theta_s` in degrees. Can be overridden by assignment;
        assigning ``None`` restores the computed value.
        """
        if self._airmass is not None :
            return self._airmass
        airmass = 1.0 / (np.cos(self.zenith_rad) +
                              0.15 * (93.885 - self.zenith_deg) ** (-1.253))
        if hasattr(airmass, "ndim"):
            airmass = np.maximum(airmass, 1)
        else:
            if airmass < 1: airmass = 1
        return airmass
    @airmass.setter
    def airmass(self, value):
        self._airmass = value


    @property
    def rayleigh_transmittance(self):
        r"""Transmittance after Rayleigh (molecular) scattering, :math:`T_r(\lambda)`.

        $$
        T_r = \exp\left[\frac{-M}{\lambda^4\,(115.6406 - 1.335/\lambda^2)}\right],
        $$

        with :math:`\lambda` in um (Leckner 1978; Bird & Riordan 1986).
        No pressure correction is applied (sea-level pressure assumed).
        """
        if hasattr(self.airmass, "ndim"):
            airmass = self.airmass[...,None]
        else:
            airmass = self.airmass
        return np.exp(-airmass / (self.wld**4 * (115.6406 - 1.335 / self.wld**2)))
    @property
    def tr(self):
        """Alias for :attr:`rayleigh_transmittance`."""
        return self.rayleigh_transmittance

    @property
    def aerosol_transmittance(self):
        r"""Transmittance after aerosol extinction, :math:`T_a(\lambda)`.

        $$
        T_a = \exp\left(-\beta\,\lambda^{-\alpha} M\right),
        $$

        with :math:`\lambda` in um. Wavelengths below 500 nm use
        (:attr:`alpha1`, :attr:`beta1`) and longer wavelengths
        (:attr:`alpha2`, :attr:`beta2`), following Bird & Riordan (1986).
        """
        wld = self.wld
        if hasattr(self.airmass, "ndim"):
            airmass = self.airmass[...,None]
        else:
            airmass = self.airmass
        # AEROSOL SCATTERING AND ABSORBTION
        ta = np.zeros_like(self.tr)
        short = wld < 0.5
        ta[..., short]  = np.exp(-self.beta1 * wld[short]  ** (-self.alpha1) * airmass)
        ta[..., ~short] = np.exp(-self.beta2 * wld[~short] ** (-self.alpha2) * airmass)
        return ta
    @property
    def ta(self):
        """Alias for :attr:`aerosol_transmittance`."""
        return self.aerosol_transmittance

    @property
    def water_vapour_transmittance(self):
        r"""Transmittance after water-vapour absorption, :math:`T_w(\lambda)`.

        $$
        T_w = \exp\left[\frac{-0.3285\,a_v\,W' M}{(1 + 20.07\,a_v M)^{0.45}}\right],
        \qquad W' = W + 0.5\,(1.42 - W),
        $$

        with precipitable water :math:`W` = 2 cm and the water-vapour
        absorption coefficient :math:`a_v` from :func:`coefficients`. The
        functional form follows Leckner (1978) and Bird & Riordan (1986).
        """
        # WATER VAPOUR ABSORPTION
        w = 2.
        # av = water vapour absorption coefficient
        av = self.cfdict["av"]
        if hasattr(self.airmass, "ndim"):
            airmass = self.airmass[...,None]
        else:
            airmass = self.airmass
        return np.exp(-0.3285 * av * (w + (1.42 - w) * 0.5) *
                         airmass / (1. + 20.07 * av * airmass) ** 0.45)
    @property
    def tw(self):
        """Alias for :attr:`water_vapour_transmittance`."""
        return self.water_vapour_transmittance

    @property
    def ozone_transmittance(self):
        r"""Transmittance after ozone absorption, :math:`T_o(\lambda)`.

        $$
        T_o = \exp(-a_o\,H_o\,M_o), \qquad
        M_o = \frac{35}{\left(1224\cos^2\theta_s + 1\right)^{1/2}},
        $$

        with ozone column :math:`H_o` = 0.344 atm cm and the ozone air mass
        :math:`M_o` of Paltridge & Platt (1976).
        """
        ao  = self.cfdict["ao"]
        em0 = 35. / ((1224. * (np.cos(self.zenith_rad)) ** 2. + 1.) ** 0.5)
        if hasattr(em0, "ndim"):
            em0 = em0[...,None]
        # old model values
        # self.to = exp(-ao * 0.344 * em0)
        # self.tu = exp(-1.41 * 0.15 * airmass / (1. + 118.3 * 0.15 * airmass) ** 0.45)
        # to be corrected to these values once I have tested equivalency with current codebase
        return np.exp(-ao * 0.344 * em0)
    @property
    def to(self):
        """Alias for :attr:`ozone_transmittance`."""
        return self.ozone_transmittance
    @property
    def tu(self):
        r"""Transmittance after absorption by uniformly mixed gases, :math:`T_u`.

        $$
        T_u = \exp\left[\frac{-1.41\,a_u M}{(1 + 118.3\,a_u M)^{0.45}}\right],
        \qquad a_u = 0.3,
        $$

        (Leckner 1978; Bird & Riordan 1986). It is applied only in the bands
        selected by the ``ozone_weights``/``dif_weights`` coefficients.
        """
        return np.exp(-1.41 * 0.3 * self.airmass / (1. + 118.3 * 0.3 * self.airmass) ** 0.45)

    @property
    def air_albedo(self):
        r"""Spectral sky albedo :math:`\rho_s(\lambda)` (dimensionless).

        Fraction of upwelling light reflected back down by the atmosphere,
        used for the multiple sea-sky reflections in
        :attr:`diffuse_irradiance`:

        $$
        \rho_s = T_o T_w\left[0.5\,T_a(1 - T_r) + 0.22\,\omega_0\,T_r(1 - T_a)\right],
        $$

        with single-scattering albedo :math:`\omega_0` = 0.928 and 0.22 the
        aerosol back-scattered fraction. Frozen at construction (air mass 1.9);
        can be overridden by assignment.
        """
        if self._air_albedo is not None :
            return self._air_albedo
        ro_S = self.to * self.tw * (self.ta * (1. - self.tr) * 0.5 + self.tr * (1 - self.ta) * 0.22 * 0.928)
        ro_S = (ro_S * (1-self.cfdict["ozone_weights"]) +
                ro_S * self.tu * self.cfdict["ozone_weights"])
        return ro_S
    @property
    def ro_S(self):
        """Alias for :attr:`air_albedo`."""
        return self.air_albedo
    @air_albedo.setter
    def air_albedo(self, value):
        self._air_albedo = value


    @property
    def direct_irradiance(self):
        r"""Direct-normal spectral irradiance :math:`E_{dir}(\lambda)` (W m-2 um-1).

        $$
        E_{dir} = H_0\,T_r\,T_a\,T_w\,T_o\,[T_u],
        $$

        where :math:`H_0` is the extraterrestrial irradiance (``extspir``)
        and :math:`T_u` is applied with the ``ozone_weights``. The irradiance
        is on a surface *normal to the beam*; multiply by
        :math:`\cos\theta_s` for a horizontal surface.
        """
        dir_Ir = self.cfdict["extspir"] * self.tr * self.ta * self.tw * self.to
        dir_Ir = (dir_Ir * (1-self.cfdict["ozone_weights"]) +
                  dir_Ir * self.tu[...,None] * self.cfdict["ozone_weights"])
        return dir_Ir
    @property
    def dir_Ir(self):
        """Alias for :attr:`direct_irradiance`."""
        return self.direct_irradiance

    @property
    def diffuse_irradiance(self):
        r"""Diffuse spectral irradiance on a horizontal surface (W m-2 um-1).

        Sum of a Rayleigh-scattered component :math:`I_r`, an
        aerosol-scattered component :math:`I_a` and a component :math:`I_g`
        from multiple reflections between the sea surface and the sky:

        $$
        \begin{aligned}
        I_r &= H_0\cos\theta_s\,T_o T_w\,T_a\,(1 - T_r)\,0.5, \\
        I_a &= H_0\cos\theta_s\,T_o T_w\,T_r\,(1 - T_a)\,\omega_0 F_s, \\
        I_g &= \frac{\left[E_{dir}\cos\theta_s + C\,(I_r + I_a)\right]
                     \rho_s\,r_g}{1 - \rho_s\,r_g}, \\
        E_{dif} &= C\,(I_r + I_a) + I_g,
        \end{aligned}
        $$

        with :math:`\omega_0` = 0.928, forward-scattered fraction
        :math:`F_s` = 0.82 and sea-surface albedo :math:`r_g` = 0.05.
        :math:`C(\lambda, \theta_s)` is the Bird (1984) correction factor
        (:func:`interpolate_lut`), interpolated in zenith angle between the
        two nearest tabulated angles by inverse-distance weighting.

        The result is squeezed, so a scalar zenith angle gives an array of
        shape ``(n_wavelengths,)``.
        """
        tree = KDTree(self.cd[:,None])
        zenith_deg = np.atleast_1d(self.zenith_deg)
        dist,ij = tree.query(zenith_deg.flatten()[:,None],2)
        frac = 1 - (dist.T/np.sum(dist, axis=1)).T
        #ij = ij.reshape(zenith_deg.shape)
        corr1 = self.cfdict["correction_lut"][:,ij[:,0].reshape(zenith_deg.shape)]
        corr2 = self.cfdict["correction_lut"][:,ij[:,1].reshape(zenith_deg.shape)]
        c2 = corr1*frac[:,0].reshape(zenith_deg.shape) + corr2*frac[:,1].reshape(zenith_deg.shape)
        c2=c2.T

        extspir = self.cfdict["extspir"]
        zenith_cos = np.cos(self.zenith_rad[...,None])
        xx = extspir * zenith_cos * self.to * self.tw
        r = xx * self.ta * (1. - self.tr) * 0.5
        a = xx * self.tr * (1. - self.ta) * 0.928 * 0.82
        r = (r * (1-self.cfdict["dif_weights"]) +
             r * self.tu[...,None] * self.cfdict["dif_weights"])
        a = (a * (1-self.cfdict["dif_weights"]) +
             a * self.tu[...,None] * self.cfdict["dif_weights"])
        g = (self.dir_Ir*zenith_cos + (r + a) * c2) * self.ro_S*0.05 / (1. - 0.05*self.ro_S)
        return np.squeeze((r + a) * c2 + g)
    @property
    def dif_Ir(self):
        """Alias for :attr:`diffuse_irradiance`."""
        return self.diffuse_irradiance

    @property
    def irradiance(self):
        """Both components as a dict with keys ``"direct"`` and ``"diffuse"``."""
        return {'direct':  self.direct_irradiance,

                'diffuse': self.diffuse_irradiance}

def all(zenith_deg, wavelengths=None):
    """Clear-sky direct and diffuse spectral irradiance.

    Convenience wrapper around :class:`Irradiance`.

    Parameters
    ----------
    zenith_deg : float or array_like
        Solar zenith angle(s) (degrees).
    wavelengths : array_like, optional
        Wavelengths (nm). Defaults to :data:`config.wavelengths`.

    Returns
    -------
    dict of numpy.ndarray
        ``"direct"``
            Direct-normal irradiance (W m-2 um-1), see
            :attr:`Irradiance.direct_irradiance`.
        ``"diffuse"``
            Diffuse horizontal irradiance (W m-2 um-1), see
            :attr:`Irradiance.diffuse_irradiance`.

    Examples
    --------
    >>> import numpy as np, irradiance
    >>> ir = irradiance.all(np.float64(30.0))
    >>> ir["direct"].shape
    (61,)
    >>> ir["diffuse"].shape
    (61,)
    """
    ir = Irradiance(zenith_deg, wavelengths=wavelengths)
    return ir.irradiance
