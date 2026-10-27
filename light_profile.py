r"""Propagation of spectral irradiance through the water column.

Given a chlorophyll profile and the spectral irradiance just below the sea
surface, :func:`compute_profile` computes the diffuse attenuation coefficient
:math:`K_d(z, \lambda)` from the inherent optical properties of water,
phytoplankton and CDOM, and propagates the light downwards layer by layer
(Sathyendranath & Platt 1988; Sathyendranath et al. 1989).

References
----------
.. [1] Sathyendranath, S. & Platt, T. (1988). The spectral irradiance field
   at the surface and in the interior of the ocean: a model for applications
   in oceanography and remote sensing. *Journal of Geophysical Research*,
   93(C8), 9270-9280. https://doi.org/10.1029/JC093iC08p09270
.. [2] Sathyendranath, S., Platt, T., Caverhill, C.M., Warnock, R.E. &
   Lewis, M.R. (1989). Remote sensing of oceanic primary production:
   computations using a spectral model. *Deep Sea Research Part A*, 36(3),
   431-453. https://doi.org/10.1016/0198-0149(89)90046-0
.. [3] Loisel, H. & Morel, A. (1998). Light scattering and chlorophyll
   concentration in case 1 waters: a reexamination. *Limnology and
   Oceanography*, 43(5), 847-858. https://doi.org/10.4319/lo.1998.43.5.0847
.. [4] Ulloa, O., Sathyendranath, S. & Platt, T. (1994). Effect of the
   particle-size distribution on the backscattering ratio in seawater.
   *Applied Optics*, 33(30), 7070-7077. https://doi.org/10.1364/AO.33.007070
.. [5] Kirk, J.T.O. (2011). *Light and Photosynthesis in Aquatic Ecosystems*
   (3rd ed.). Cambridge University Press.
   https://doi.org/10.1017/CBO9781139168212
"""

import numpy as np

import config
import scatter
from absorption import calc_ac

def compute_profile(chl_profile,
                    zenith_deg,
                    direct,
                    diffuse,
                    yellow_substance=0.3,
                    wavelengths=None,
                    depths=None):
    r"""Compute the spectral light field as a function of depth.

    Parameters
    ----------
    chl_profile : numpy.ndarray
        Chlorophyll-a profile (mg m-3), shape ``(n_depths,)``, on the grid
        ``depths`` (e.g. from :func:`chlorophyll.sinusodal_profile`).
        Values must be positive (``log10`` is taken).
    zenith_deg : float
        Solar zenith angle in air (degrees).
    direct : numpy.ndarray
        Direct spectral irradiance just below the surface, shape
        ``(n_wavelengths,)`` (e.g. ``surface_light.instant(...)["direct"]``,
        einstein m-2 h-1 nm-1).
    diffuse : numpy.ndarray
        Diffuse spectral irradiance just below the surface, same shape and
        units as ``direct``.
    yellow_substance : float, default 0.3
        Ratio :math:`a_y(440)/a_{ph}(440)` of CDOM to phytoplankton absorption
        at 440 nm. ``None`` uses :data:`config.yellow_substance`.
    wavelengths : numpy.ndarray, optional
        Wavelengths (nm) matching ``direct``/``diffuse``. Defaults to
        :data:`config.wavelengths`.
    depths : numpy.ndarray, optional
        Depth levels (m) matching ``chl_profile``. Defaults to
        :data:`config.depths`.

    Returns
    -------
    dict
        ``"Iz"`` : numpy.ndarray, shape (n_depths, n_wavelengths)
            Downwelling spectral irradiance :math:`E_d(z, \lambda)`, in the
            units of ``direct``.
        ``"IzPAR"`` : numpy.ndarray, shape (n_depths,)
            Spectrally integrated irradiance (PAR) at each depth
            (:math:`\sum_\lambda E_d\,\Delta\lambda`).
        ``"K"`` : numpy.ndarray, shape (n_depths, n_wavelengths)
            Diffuse attenuation coefficient :math:`K_d(z, \lambda)` (m-1).
        ``"mu_d"`` : numpy.ndarray, shape (n_wavelengths,)
            Mean cosine of the downwelling light just below the surface.
        ``"euphotic_depth"`` : float
            Deepest grid depth (m) at which PAR is still at least 1 % of
            the PAR at the first depth level.
        ``"atot"`` : numpy.ndarray, shape (n_depths, n_wavelengths)
            Total absorption coefficient :math:`a(z, \lambda)` (m-1).

    Notes
    -----
    **Mean cosine.** The direct beam is refracted to the in-water angle
    :math:`\theta_w = \arcsin(\sin\theta_s/1.333)`; diffuse light is assigned
    a mean cosine of 0.831 (Sathyendranath & Platt 1988):

    $$
    \mu_d(\lambda) = \frac{E_{dir}\cos\theta_w + 0.831\,E_{dif}}{E_{dir} + E_{dif}}.
    $$

    **Absorption.**

    $$
    a = a_w + a_{ph} + a_y(440)\,e^{-0.014(\lambda - 440)} + 2\,b_{br},
    \qquad a_y(440) = Y\,a_{ph}(440),
    $$

    with :math:`a_w` from :func:`scatter.calc_aw`, :math:`a_{ph}` from
    :func:`absorption.calc_ac`, :math:`Y` = ``yellow_substance`` and the
    Raman term :math:`b_{br}` (:func:`scatter.calc_bbr`) treated as a loss.

    **Backscattering.** Particle scattering at 660 nm follows Loisel & Morel
    (1998), with a chlorophyll-dependent spectral slope; the backscattering
    ratio follows Ulloa et al. (1994), bounded to [0.0005, 0.01]:

    $$
    b_p(\lambda) = 0.407\,B^{0.795}\left(\frac{660}{\lambda}\right)^{-\log_{10}B},
    \qquad
    \tilde b_{b} = 0.0078 - 0.0042\log_{10}B,
    $$

    $$
    b_b = \tilde b_b\,b_p + 0.5\,b_w.
    $$

    **Attenuation and propagation.**

    $$
    K_d(z,\lambda) = \frac{a + b_b}{\mu_d}, \qquad
    E_d(z_i) = E_d(z_{i-1})\exp\left[-K_d(z_i)\,\Delta z_i\right],
    $$

    starting from :math:`E_d(0^-) = E_{dir} + E_{dif}` and using
    ``numpy.gradient(depths)`` as the layer thickness :math:`\Delta z`, so the
    first level is already attenuated over half a layer. :math:`\mu_d` is
    kept constant with depth.

    References
    ----------
    .. [1] Sathyendranath & Platt (1988), https://doi.org/10.1029/JC093iC08p09270
    .. [2] Loisel & Morel (1998), https://doi.org/10.4319/lo.1998.43.5.0847
    .. [3] Ulloa et al. (1994), https://doi.org/10.1364/AO.33.007070

    Examples
    --------
    >>> import pandas as pd
    >>> import chlorophyll, surface_light, light_profile
    >>> chlz = chlorophyll.sinusodal_profile(1.5, rho=0.1)
    >>> irs = surface_light.instant(pd.Timestamp("2020-05-06 12:00"), 45, 25)
    >>> irz = light_profile.compute_profile(chlz, 28, irs["direct"], irs["diffuse"])
    >>> irz["Iz"].shape
    (500, 61)
    """
    depths = config.depths if depths is None else depths
    wavelengths = config.wavelengths if wavelengths is None else wavelengths
    yellow_substance = config.yellow_substance if yellow_substance is None else yellow_substance
    scdict = scatter.all(wavelengths=wavelengths)

    zenw = np.arcsin(np.sin(np.deg2rad(zenith_deg)) / 1.333)

    I0 = direct + diffuse
    mu_d = (direct * np.cos(zenw) + diffuse * 0.831000) / I0

    ac,_ = calc_ac(chl_profile, wavelengths)
    ac440idx = np.argmin(np.abs(wavelengths-440))
    ac440 = ac[:,ac440idx]
    ay440 = yellow_substance * ac440
    atot = scdict["aw"] + ac + (ay440[:,np.newaxis] * scdict["ay"]) + 2.0 * scdict["bbr"]


    power = -np.log10(chl_profile)
    bc660 = 0.407 * chl_profile**0.795
    bbtilda = (0.78 + 0.42 * power) * 0.01
    bbtilda[bbtilda < 0.0005] = 0.0005
    bbtilda[bbtilda > 0.01] = 0.01

    bc = bc660[:,np.newaxis] * (660.0 / wavelengths)**power[:,np.newaxis]
    bc[bc < 0]=0.0
    bb = bc * bbtilda[:,np.newaxis] + scdict["bw"] * 0.50
    K = (atot + bb) / mu_d

    Izlist = [I0,]
    for dz, Ki in zip(np.gradient(depths), K):
        Izlist.append(Izlist[-1] * np.exp(-Ki * dz))
    Iz = np.array(Izlist)[1:,:]
    izpar=(Iz*scdict["d_lambda"]).sum(axis=1)
    euphotic_depth = depths[np.nonzero(izpar/izpar[0]>=0.01)[0].max()]
    return dict(Iz=Iz, IzPAR=izpar, K=K, mu_d=mu_d,
                euphotic_depth=euphotic_depth, atot=atot)

    x=np.log10(100*izpar/izpar[0])[::-1] #need to reverse arrays as np.interp cannot handle decreasing values of x :(
    y=depth_array[::-1]
    euphotic_depth=np.interp(0,x,y)
    euph_id=np.where(depth_array<euphotic_depth)[0][-1]
    return izpar, Iz, K, euphotic_depth, euph_id, ac