"""Vertical structure of chlorophyll-a.

Ocean-colour satellites see chlorophyll-a only in the upper part of the water
column (roughly the first optical depth). The model extends this surface
value to a full depth profile using a Gaussian deep chlorophyll maximum on top
of a uniform background (Lewis et al. 1983; Platt et al. 1988).

References
----------
.. [1] Lewis, M.R., Cullen, J.J. & Platt, T. (1983). Phytoplankton and thermal
   structure in the upper ocean: consequences of nonuniformity in chlorophyll
   profile. *Journal of Geophysical Research*, 88(C4), 2565-2570.
   https://doi.org/10.1029/JC088iC04p02565
.. [2] Platt, T., Sathyendranath, S., Caverhill, C.M. & Lewis, M.R. (1988).
   Ocean primary production and available light: further algorithms for
   remote sensing. *Deep Sea Research Part A*, 35(6), 855-879.
   https://doi.org/10.1016/0198-0149(88)90064-7
.. [3] Longhurst, A., Sathyendranath, S., Platt, T. & Caverhill, C. (1995).
   An estimate of global primary production in the ocean from satellite
   radiometer data. *Journal of Plankton Research*, 17(6), 1245-1271.
   https://doi.org/10.1093/plankt/17.6.1245
"""

import numpy as np

import config

def sinusodal_profile(chl, sigma=24, rho=0.86, z_m=60, depths=None, mld=None):
    r"""Generate a chlorophyll depth profile with a deep maximum.

    Despite the function name, the profile is a *shifted Gaussian*: a uniform
    background plus a Gaussian peak (the deep chlorophyll maximum). The
    profile is scaled so that its value at the surface equals ``chl``.

    Parameters
    ----------
    chl : float or array_like
        Surface chlorophyll-a concentration :math:`B(0)` (mg m-3), typically
        from satellite ocean colour. Arrays of any shape are accepted and
        produce one profile per element.
    sigma : float, default 24
        Width (standard deviation) of the Gaussian peak, :math:`\sigma` (m).
    rho : float, default 0.86
        Ratio of the Gaussian peak height to the total chlorophyll at the
        depth of the maximum, :math:`\rho` (dimensionless, 0 <= rho < 1).
        ``rho -> 0`` gives a vertically uniform profile.
    z_m : float, default 60
        Depth of the chlorophyll maximum, :math:`z_m` (m).
    depths : array_like, optional
        Depth levels (m), positive downwards. Defaults to :data:`config.depths`.
    mld : float, optional
        If given, only depths ``<= mld`` (mixed-layer depth, m) are returned.

    Returns
    -------
    numpy.ndarray
        Chlorophyll-a concentration :math:`B(z)` (mg m-3). Shape
        ``(n_depths,)`` for scalar ``chl``, otherwise
        ``chl.shape + (n_depths,)``.

    Notes
    -----
    The shifted Gaussian of Lewis et al. (1983) and Platt et al. (1988) is

    $$
    B(z) = B_0 + \frac{h}{\sigma\sqrt{2\pi}}
           \exp\left[-\frac{(z - z_m)^2}{2\sigma^2}\right],
    $$

    where :math:`B_0` is the background concentration (mg m-3) and
    :math:`h` the integrated biomass in the peak (mg m-2). Here the peak is
    described by :math:`\rho = H/(H + B_0)`, with :math:`H = h/(\sigma\sqrt{2\pi})`
    the peak height, and :math:`B_0` is chosen so that :math:`B(0)` equals
    the input ``chl``:

    $$
    B_0 = \frac{B(0)}{1 + \dfrac{\rho}{1-\rho}\exp\left(-\dfrac{z_m^2}{2\sigma^2}\right)},
    \qquad
    h = \sigma\,\frac{\rho}{1-\rho}\,B_0\sqrt{2\pi}.
    $$

    The defaults (``sigma=24``, ``rho=0.86``, ``z_m=60``) describe a
    pronounced subsurface maximum typical of stratified waters. Region- and
    season-specific parameter sets are given by Longhurst et al. (1995) for
    their biogeochemical provinces.

    Levels where the Gaussian exponent exceeds 675 in magnitude (more than
    ~37 sigma from the peak) are set to zero to avoid floating-point
    underflow.

    References
    ----------
    .. [1] Lewis et al. (1983), https://doi.org/10.1029/JC088iC04p02565
    .. [2] Platt et al. (1988), https://doi.org/10.1016/0198-0149(88)90064-7
    .. [3] Longhurst et al. (1995), https://doi.org/10.1093/plankt/17.6.1245

    Examples
    --------
    >>> import chlorophyll
    >>> chlz = chlorophyll.sinusodal_profile(1.5, rho=0.1)
    >>> chlz.shape
    (500,)
    >>> round(float(chlz[0]), 6)
    1.5
    """
    depths = config.depths if depths is None else depths
    if mld is not None:
        depths = depths[depths<=mld]
    B_0 = chl / (1 + (rho / (1 - rho)) * np.exp(-z_m ** 2 / (2 * sigma ** 2)))
    h =  sigma * (rho / (1 - rho)) * B_0 * np.sqrt(2 * np.pi)
    gauss_height = h / (sigma * np.sqrt(2*np.pi))

    c_arr = -0.5 * ( (depths - z_m) / sigma ) ** 2
    if np.iterable(chl):
        chl_profile = gauss_height[...,None] * np.exp(c_arr) + B_0[...,None]
        chl_profile[...,np.abs(c_arr) > 675.] = 0
    else:
        chl_profile = gauss_height * np.exp(c_arr) + B_0
        chl_profile[(np.abs(c_arr) > 675.)] = 0
    return chl_profile
