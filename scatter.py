r"""Inherent optical properties of pure seawater and CDOM.

Spectral absorption and scattering of the non-phytoplankton constituents of
the water column, used by :func:`light_profile.compute_profile` to build the
total attenuation:

* pure-water absorption :math:`a_w(\lambda)` (Pope & Fry 1997),
* pure-seawater scattering :math:`b_w(\lambda)` (Morel 1974),
* Raman scattering :math:`b_r(\lambda)` (Bartlett et al. 1998),
* the spectral shape of absorption by coloured dissolved organic matter
  (CDOM, "yellow substance") (Bricaud et al. 1981).

Attributes
----------
bw500 : float
    Scattering coefficient of pure seawater at 500 nm (m-1), Morel (1974).
br488 : float
    Raman scattering coefficient of pure water at 488 nm (m-1),
    Bartlett et al. (1998).

References
----------
.. [1] Pope, R.M. & Fry, E.S. (1997). Absorption spectrum (380-700 nm) of
   pure water. II. Integrating cavity measurements. *Applied Optics*,
   36(33), 8710-8723. https://doi.org/10.1364/AO.36.008710
.. [2] Morel, A. (1974). Optical properties of pure water and pure sea water.
   In N.G. Jerlov & E. Steemann Nielsen (Eds.), *Optical Aspects of
   Oceanography* (pp. 1-24). Academic Press, London.
.. [3] Bartlett, J.S., Voss, K.J., Sathyendranath, S. & Vodacek, A. (1998).
   Raman scattering by pure water and seawater. *Applied Optics*, 37(15),
   3324-3332. https://doi.org/10.1364/AO.37.003324
.. [4] Bricaud, A., Morel, A. & Prieur, L. (1981). Absorption by dissolved
   organic matter of the sea (yellow substance) in the UV and visible
   domains. *Limnology and Oceanography*, 26(1), 43-53.
   https://doi.org/10.4319/lo.1981.26.1.0043
"""

import numpy as np

import config

bw500 = 0.00288  # Scattering coefficient of pure sea-water at 500nm
br488 = 0.00027

def all(wavelengths=None):
    """Collect all non-phytoplankton optical properties on one wavelength grid.

    Parameters
    ----------
    wavelengths : array_like, optional
        Wavelengths (nm). Defaults to :data:`config.wavelengths`.

    Returns
    -------
    dict of numpy.ndarray
        Dictionary with one array of shape ``(n_wavelengths,)`` per key:

        ``"ay"``
            Normalised CDOM absorption spectrum, equal to 1 at 440 nm
            (dimensionless); see :func:`calc_ay`.
        ``"bbr"``
            Raman (back)scattering coefficient (m-1); see :func:`calc_bbr`.
        ``"bw"``
            Pure-seawater scattering coefficient (m-1); see :func:`calc_bw`.
        ``"aw"``
            Pure-water absorption coefficient (m-1); see :func:`calc_aw`.
        ``"d_lambda"``
            Width of each wavelength band (nm), used as the quadrature
            weight when integrating over the spectrum. The last band is
            given the width of the second-to-last.

    Examples
    --------
    >>> import scatter
    >>> sc = scatter.all()
    >>> sorted(sc)
    ['aw', 'ay', 'bbr', 'bw', 'd_lambda']
    >>> float(sc["d_lambda"].sum())
    305.0
    """
    wavelengths = config.wavelengths if wavelengths is None else wavelengths
    return dict(
        ay = calc_ay(wavelengths),
        bbr = calc_bbr(wavelengths),
        bw = calc_bw(wavelengths),
        aw = calc_aw(wavelengths),
        d_lambda = np.append(np.diff(wavelengths), np.diff(wavelengths)[-1]))

def calc_bw(wavelengths=None):
    r"""Compute the scattering coefficient of pure seawater.

    Parameters
    ----------
    wavelengths : array_like, optional
        Wavelengths (nm). Defaults to :data:`config.wavelengths`.

    Returns
    -------
    numpy.ndarray
        Total scattering coefficient of pure seawater :math:`b_w` (m-1).

    Notes
    -----
    Molecular (density-fluctuation) scattering follows a power law in
    wavelength [1]_:

    $$
    b_w(\lambda) = b_w(500)\left(\frac{\lambda}{500}\right)^{-4.3},
    $$

    with :math:`b_w(500)` = :data:`bw500` = 0.00288 m-1. Half of it
    (:math:`b_{bw} = 0.5\,b_w`) is backscattered, which is how it enters
    :func:`light_profile.compute_profile`.

    References
    ----------
    .. [1] Morel, A. (1974). Optical properties of pure water and pure sea
       water. In *Optical Aspects of Oceanography*, Academic Press, 1-24.
    """
    wavelengths = config.wavelengths if wavelengths is None else wavelengths
    bw = bw500 * (wavelengths / 500.) ** -4.3
    return bw

def calc_bbr(wavelengths=None):
    r"""Compute Raman backscattering in pure seawater.

    Parameters
    ----------
    wavelengths : array_like, optional
        Wavelengths (nm). Defaults to :data:`config.wavelengths`.

    Returns
    -------
    numpy.ndarray
        Raman backscattering coefficient :math:`b_{br}` (m-1).

    Notes
    -----
    Raman (inelastic) scattering by water molecules is isotropic, so half of
    it is backscattered. With the Raman scattering coefficient at 488 nm
    :data:`br488` = 2.7e-4 m-1 and a :math:`\lambda^{-5.3}` dependence on
    excitation wavelength [1]_:

    $$
    b_{br}(\lambda) = 0.5\,b_r(488)\left(\frac{\lambda}{488}\right)^{-5.3}.
    $$

    In this model Raman scattering is treated as a loss term only: it is
    added (twice) to the total absorption in
    :func:`light_profile.compute_profile`, and the light re-emitted at longer
    wavelengths is not tracked.

    References
    ----------
    .. [1] Bartlett, J.S., Voss, K.J., Sathyendranath, S. & Vodacek, A.
       (1998). Raman scattering by pure water and seawater. *Applied Optics*,
       37(15), 3324-3332. https://doi.org/10.1364/AO.37.003324
    """
    wavelengths = config.wavelengths if wavelengths is None else wavelengths
    bbr = 0.5 * br488 * (wavelengths/ 488.) ** -5.3
    return bbr

def calc_ay(wavelengths=None):
    r"""Compute the normalised absorption spectrum of CDOM.

    Parameters
    ----------
    wavelengths : array_like, optional
        Wavelengths (nm). Defaults to :data:`config.wavelengths`.

    Returns
    -------
    numpy.ndarray
        CDOM absorption normalised to 440 nm, :math:`a_y(\lambda)/a_y(440)`
        (dimensionless).

    Notes
    -----
    CDOM (yellow substance) absorption decreases exponentially with
    wavelength [1]_:

    $$
    \frac{a_y(\lambda)}{a_y(440)} = \exp\left[-S\,(\lambda - 440)\right],
    \qquad S = 0.014\ \mathrm{nm^{-1}}.
    $$

    The magnitude :math:`a_y(440)` is set in
    :func:`light_profile.compute_profile` as a fixed fraction
    (``yellow_substance``) of phytoplankton absorption at 440 nm, following
    the Case-1 assumption that CDOM co-varies with phytoplankton
    (Prieur & Sathyendranath 1981).

    References
    ----------
    .. [1] Bricaud, A., Morel, A. & Prieur, L. (1981). Absorption by
       dissolved organic matter of the sea (yellow substance) in the UV and
       visible domains. *Limnology and Oceanography*, 26(1), 43-53.
       https://doi.org/10.4319/lo.1981.26.1.0043
    .. [2] Prieur, L. & Sathyendranath, S. (1981). An optical classification
       of coastal and oceanic waters based on the specific spectral
       absorption curves of phytoplankton pigments, dissolved organic matter,
       and other particulate materials. *Limnology and Oceanography*, 26(4),
       671-689. https://doi.org/10.4319/lo.1981.26.4.0671
    """
    wavelengths = config.wavelengths if wavelengths is None else wavelengths
    ay = np.exp(-0.014 * (wavelengths - 440))
    return ay

def calc_aw(wavelengths=None):
    """Return the absorption spectrum of pure water.

    Parameters
    ----------
    wavelengths : array_like, optional
        Wavelengths (nm). If omitted, the tabulated values on the native
        grid :data:`config.phyto_wl` (400-700 nm, 5 nm) are returned
        unchanged; otherwise they are linearly interpolated (and held
        constant outside 400-700 nm).

    Returns
    -------
    numpy.ndarray
        Pure-water absorption coefficient :math:`a_w` (m-1).

    Notes
    -----
    Values are the integrating-cavity measurements of Pope & Fry (1997) [1]_,
    sampled at 5 nm. Note that, unlike the other functions in this module,
    the default grid is :data:`config.phyto_wl` rather than
    :data:`config.wavelengths` (the two are identical by default).

    References
    ----------
    .. [1] Pope, R.M. & Fry, E.S. (1997). Absorption spectrum (380-700 nm) of
       pure water. II. Integrating cavity measurements. *Applied Optics*,
       36(33), 8710-8723. https://doi.org/10.1364/AO.36.008710

    Examples
    --------
    >>> import numpy as np
    >>> from scatter import calc_aw
    >>> float(calc_aw(np.array([440.0]))[0])
    0.00635
    """
    aw = np.asarray(
        [0.00663, 0.00530, 0.00473, 0.00444, 0.00454, 0.00478, 0.00495, 0.00530,
         0.00635, 0.00751, 0.00922, 0.00962, 0.00979, 0.01011, 0.0106,  0.0114,
         0.0127,  0.0136,  0.0150,  0.0173,  0.0204,  0.0256,  0.0325,  0.0396,
         0.0409,  0.0417,  0.0434,  0.0452,  0.0474,  0.0511,  0.0565,  0.0596,
         0.0619,  0.0642,  0.0695,  0.0772,  0.0896,  0.1100,  0.1351,  0.1672,
         0.2224,  0.2577,  0.2644,  0.2678,  0.2755,  0.2834,  0.2916,  0.3012,
         0.3108,  0.325,   0.340,   0.371,   0.410,   0.429,   0.439,   0.448,
         0.465,   0.486,   0.516,   0.559,   0.624])
    if wavelengths is None:
        return aw
    else:
        return np.interp(wavelengths, config.phyto_wl, aw)
