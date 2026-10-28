r"""Spectral irradiance just below the sea surface at a given instant.

Combines the solar geometry (:mod:`daylight`), the clear-sky spectral model
(:mod:`irradiance`), a cloud correction and Fresnel reflection at the sea
surface, and finally scales the spectrum so that its integral follows a
sinusoidal diurnal cycle constrained by the daily PAR observed by satellite
(Platt & Sathyendranath 1988; Sathyendranath & Platt 1988).

References
----------
.. [1] Platt, T. & Sathyendranath, S. (1988). Oceanic primary production:
   estimation by remote sensing at local and regional scales. *Science*,
   241(4873), 1613-1620. https://doi.org/10.1126/science.241.4873.1613
.. [2] Sathyendranath, S. & Platt, T. (1988). The spectral irradiance field
   at the surface and in the interior of the ocean: a model for applications
   in oceanography and remote sensing. *Journal of Geophysical Research*,
   93(C8), 9270-9280. https://doi.org/10.1029/JC093iC08p09270
.. [3] Morel, A. & Smith, R.C. (1974). Relation between total quanta and
   total energy for aquatic photosynthesis. *Limnology and Oceanography*,
   19(4), 591-600. https://doi.org/10.4319/lo.1974.19.4.0591
.. [4] Kirk, J.T.O. (2011). *Light and Photosynthesis in Aquatic Ecosystems*
   (3rd ed.). Cambridge University Press.
   https://doi.org/10.1017/CBO9781139168212
"""


import numpy as np
import pandas as pd

import daylight
import scatter
import config

import irradiance as bird_irr

def instant(dtm, lat, par, wavelengths=None, cloud=0, verbose=False):
    r"""Spectral direct and diffuse irradiance just below the surface.

    Parameters
    ----------
    dtm : datetime-like
        Date and local solar time (a single timestamp). Outside daylight
        (sun below the horizon, or before sunrise / after sunset of the
        sinusoidal PAR cycle) all irradiances are zero.
    lat : float
        Latitude (degrees north).
    par : float
        Daily photosynthetically available radiation above the surface,
        e.g. the satellite PAR product (einstein m-2 d-1).
    wavelengths : numpy.ndarray, optional
        Wavelengths (nm). Defaults to :data:`config.wavelengths`.
    cloud : float, default 0
        Cloud cover (percent, 0-100).
    verbose : bool, default False
        Print intermediate quantities (day length, noon irradiance,
        reflection losses, scaling factor).

    Returns
    -------
    dict
        ``"direct"`` : numpy.ndarray, shape (n_wavelengths,)
            Direct spectral irradiance on a horizontal surface just below
            the sea surface (einstein m-2 h-1 nm-1).
        ``"diffuse"`` : numpy.ndarray, shape (n_wavelengths,)
            Diffuse spectral irradiance just below the sea surface
            (einstein m-2 h-1 nm-1).
        ``"total"`` : numpy.ndarray, shape (n_wavelengths,)
            ``direct + diffuse``.
        ``"i_zero"`` : float
            Spectrally integrated total irradiance just below the surface
            (einstein m-2 h-1).

    Notes
    -----
    The calculation proceeds in six steps.

    **1. Noon irradiance.** From the day length :math:`D` (h, see
    :func:`daylight.daylength`) the maximum (noon) PAR above the surface is
    :math:`I_m = \pi\,\mathrm{PAR}/(2D)`, which follows from assuming a
    sinusoidal diurnal cycle whose integral is the daily PAR.

    **2. Clear-sky spectrum.** Direct and diffuse spectra are computed with
    :func:`irradiance.all` at the solar zenith angle :math:`\theta_s` and
    scaled by the top-of-atmosphere irradiance relative to the solar
    constant (``TOA_Irr / 1353``).

    **3. Clouds.** With cloud fraction :math:`c` the direct component is
    reduced by :math:`(1 - c)` and the diffuse component is set so that the
    total equals the clear-sky total times

    $$
    F = \frac{(1 - 0.5c)\,\left[0.82 - \alpha(1 - c)\right]}{0.82 - \alpha},
    \qquad \alpha = \frac{0.28}{1 + 6.43\cos\theta_s}.
    $$

    **4. Units.** Energy is converted to quanta (W m-2 um-1 to
    einstein m-2 h-1 nm-1) using :math:`E = hc/\lambda` and Avogadro's number
    (Morel & Smith 1974); the direct beam is projected onto the horizontal
    by :math:`\cos\theta_s`.

    **5. Surface reflection.** The direct beam loses the unpolarised Fresnel
    reflectance

    $$
    r = \frac{1}{2}\frac{\sin^2(\theta_s - \theta_w)}{\sin^2(\theta_s + \theta_w)}
      + \frac{1}{2}\frac{\tan^2(\theta_s - \theta_w)}{\tan^2(\theta_s + \theta_w)},
    \qquad \sin\theta_w = \frac{\sin\theta_s}{1.333},
    $$

    and the diffuse light a fixed 5.5 % (transmission 0.945). At normal
    incidence (:math:`\theta_s = 0`) the limit
    :math:`r = ((n-1)/(n+1))^2 \approx 0.020` is used.

    **6. Scaling to satellite PAR.** The instantaneous above-surface
    irradiance is taken from the sinusoidal cycle,
    :math:`I(t) = I_m \sin(\pi\,(t - t_{\mathrm{rise}})/D)`, and both
    below-surface components are multiplied by :math:`I(t)` divided by the
    spectrally integrated above-surface model irradiance. The model thus
    supplies only the *spectral shape* and the direct/diffuse partition,
    while the magnitude comes from the observed PAR.

    References
    ----------
    .. [1] Platt & Sathyendranath (1988),
       https://doi.org/10.1126/science.241.4873.1613
    .. [2] Sathyendranath & Platt (1988), https://doi.org/10.1029/JC093iC08p09270
    .. [3] Morel & Smith (1974), https://doi.org/10.4319/lo.1974.19.4.0591

    Examples
    --------
    >>> import pandas as pd, surface_light
    >>> irs = surface_light.instant(pd.Timestamp("2020-05-06 12:00"), 45, 25)
    >>> sorted(irs)
    ['diffuse', 'direct', 'i_zero', 'total']
    >>> irs["total"].shape
    (61,)
    >>> surface_light.instant(pd.Timestamp("2020-05-06 02:00"), 45, 25)["i_zero"]
    0.0
    """
    dtm = pd.to_datetime(dtm)
    wavelengths = config.wavelengths if wavelengths is None else wavelengths
    scdict = scatter.all(wavelengths=wavelengths)
 
    #Calculate daylengths etc
    daydict = daylight.all(dtm, lat)
    zenithR = daydict["zenith_rad"]

    # Fraction of the daylight period elapsed; no light outside (0, 1) or
    # with the sun below the horizon
    frac_of_day = 0.0
    if daydict["daylength"] > 0:
        frac_of_day = ((dtm - daydict["sunrise"]).total_seconds()/3600)/daydict["daylength"]
    if not (0 < frac_of_day < 1) or zenithR >= np.pi / 2:
        dark = np.zeros(np.shape(wavelengths))
        return {"direct": dark, "diffuse": dark.copy(), "total": dark.copy(), "i_zero": 0.0}

    iom = par * np.pi / (2 * daydict["daylength"])
    if verbose:
        print('Daylength: ', daydict["daylength"])
        print('iom: ', iom)

   
    #Calculate bird irridiance
    components = bird_irr.all(zenith_deg=np.rad2deg(zenithR), wavelengths=wavelengths)
    direct = components['direct']
    diffuse = components['diffuse']
    
    # correct for seasonal variation in solar energy
    sol = daydict["TOA_Irr"]
    direct  = direct * sol / 1353.
    diffuse = diffuse * sol / 1353.
    if verbose:
        print("direct:", direct)
        print("diffuse:", diffuse)
    i_direct  =  np.sum(direct * np.cos(zenithR) * scdict["d_lambda"])
    i_diffuse =  np.sum(diffuse * scdict["d_lambda"])
    if verbose:
        print("i_direct:", i_direct)
        print("i_diffuse:", i_diffuse)
    i_surface = i_direct + i_diffuse

    # compute and correct for albedo
    albedo = 0.28 / (1 + 6.43 * np.cos(zenithR))
    cc = cloud / 100.
    idir1 = i_direct * (1 - cc)
    flux = (((1 - 0.5 * cc) * (0.82 - albedo * (1 - cc)) * np.cos(zenithR)) /
            ((0.82 - albedo) * np.cos(zenithR)))
    idif1 = i_surface * flux - idir1
    direct = direct * (idir1 / i_direct) 
    diffuse = diffuse * (idif1 / i_diffuse)

    # calculate reflection and convert watts/micron to einsteins/hr/nm
    zenithW = np.arcsin(np.sin(zenithR) / 1.333)
    if zenithR > 0:
        ref = 0.5 * (np.sin(zenithR - zenithW)) ** 2 / (np.sin(zenithR + zenithW)) ** 2
        ref += 0.5 * (np.tan(zenithR - zenithW)) ** 2 / (np.tan(zenithR + zenithW)) ** 2
    else:
        # Normal-incidence limit of the Fresnel equations
        ref = ((1.333 - 1) / (1.333 + 1)) ** 2

    # compute surface irradiance across spectrum
    wl_coeff = wavelengths * 36. / (19.87 * 6.022 * 10 ** 7)
    direct = direct * wl_coeff * np.cos(zenithR)
    diffuse = diffuse * wl_coeff

    i_surface = (direct*scdict["d_lambda"]).sum() + (diffuse*scdict["d_lambda"]).sum()
    direct = direct * (1 - ref)
    diffuse = diffuse * 0.945

    total = direct + diffuse
    reflec_loss = i_surface-(total*scdict["d_lambda"]).sum()
    if verbose:
        print('loss due to surf ref:', reflec_loss)
        print('% loss due to surf ref:', 100*(i_surface-(total*scdict["d_lambda"]).sum())/i_surface)
        print('Total I: ', (total*scdict["d_lambda"]).sum())

    # Calculate surface irradiance from total daily surface irradiance
    # (e.g. satellite PAR)
    i_surface_par = iom * np.sin(np.pi * frac_of_day)
        
    if verbose:
        print('Isurface : ', i_surface.sum())
    # Adjustment to the difuse and direct component: from use of measured total daily surface irradiance (
    # e.g. satellite PAR) to compute the surface irradiance at all time. SSP
    adjustment = i_surface_par / i_surface
    if verbose:
        print('adjustment: ', adjustment)
    # compute irradiance surface value
    direct = direct * adjustment
    diffuse = diffuse * adjustment
    total = direct + diffuse
        
    i_zero = (total*scdict["d_lambda"]).sum()
    if verbose:
        print('I_Z: ', i_zero)
    return {"direct":direct, "diffuse":diffuse, "total": total, "i_zero":i_zero}

