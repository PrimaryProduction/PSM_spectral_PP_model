"""Partitioning of total chlorophyll-a into phytoplankton size classes.

Implements the three-component model of Brewin et al. (2010), which splits
total chlorophyll-a into picoplankton (< 2 um), nanoplankton (2-20 um) and
microplankton (> 20 um) fractions. The split is used by :mod:`absorption`
to build the phytoplankton absorption spectrum.

Attributes
----------
cm_pn : float
    Asymptotic maximum chlorophyll of the combined pico + nano fraction,
    :math:`C^m_{p,n}` (mg m-3).
s_pn : float
    Initial slope of the pico + nano fraction, :math:`S_{p,n} = D_{p,n}/C^m_{p,n}`
    ((mg m-3)-1), where :math:`D_{p,n} = 0.94` is the pico + nano fraction
    of total chlorophyll as total chlorophyll tends to zero.
cm_p : float
    Asymptotic maximum chlorophyll of the pico fraction, :math:`C^m_p` (mg m-3).
s_p : float
    Initial slope of the pico fraction, :math:`S_p = D_p/C^m_p` ((mg m-3)-1),
    with :math:`D_p = 0.80`.

References
----------
.. [1] Brewin, R.J.W., Sathyendranath, S., Hirata, T., Lavender, S.J.,
   Barciela, R.M. & Hardman-Mountford, N.J. (2010). A three-component model
   of phytoplankton size class for the Atlantic Ocean. *Ecological Modelling*,
   221(11), 1472-1483. https://doi.org/10.1016/j.ecolmodel.2010.02.014
.. [2] Brewin, R.J.W., Sathyendranath, S., Jackson, T., Barlow, R.,
   Brotas, V., Airs, R. & Lamont, T. (2015). Influence of light in the mixed
   layer on the parameters of a three-component model of phytoplankton size
   class. *Journal of Geophysical Research: Oceans*, 120.
   https://doi.org/10.1002/2014JC009859
"""

import numpy as np

from config import phyto_wl

cm_pn = 0.77
s_pn = 0.94 / cm_pn
cm_p = 0.13
s_p = 0.80 / cm_p

def brewin(chl):
    r"""Calculate Chl size fractions for different phytoplankton size classes.

    Parameters
    ----------
    chl : float or array_like
        Total chlorophyll-a concentration :math:`C` (mg m-3). May be a scalar
        or an array of any shape (for example a depth profile from
        :func:`chlorophyll.sinusodal_profile`).

    Returns
    -------
    pico_chl : float or numpy.ndarray
        Chlorophyll in picoplankton, :math:`C_p` (mg m-3).
    nano_chl : float or numpy.ndarray
        Chlorophyll in nanoplankton, :math:`C_n` (mg m-3).
    micro_chl : float or numpy.ndarray
        Chlorophyll in microplankton, :math:`C_m` (mg m-3).

        Scalars are returned for scalar input, otherwise arrays with the
        shape of ``chl``.

    Notes
    -----
    The combined pico + nano and the pico fractions saturate exponentially
    with total chlorophyll [1]_:

    $$
    C_{p,n} = C^m_{p,n}\left[1 - \exp(-S_{p,n} C)\right], \qquad
    C_p = C^m_p\left[1 - \exp(-S_p C)\right],
    $$

    and the nano and micro fractions follow by difference,
    :math:`C_n = C_{p,n} - C_p` and :math:`C_m = C - C_{p,n}`. Negative
    fractions are set to zero. The parameters are the module-level constants
    :data:`cm_pn`, :data:`s_pn`, :data:`cm_p` and :data:`s_p` (see [2]_ for how
    these parameters vary with the light environment).

    The three fractions add up to ``chl`` wherever none of them is clipped.

    References
    ----------
    .. [1] Brewin et al. (2010), https://doi.org/10.1016/j.ecolmodel.2010.02.014
    .. [2] Brewin et al. (2015), https://doi.org/10.1002/2014JC009859

    Examples
    --------
    >>> from size_fraction import brewin
    >>> pico, nano, micro = brewin(0.3)
    >>> round(pico + nano + micro, 6)
    0.3
    """
    #compute fractions
    pico_chl  = cm_p * (1.0 - np.exp(-s_p * chl))
    nano_chl  = cm_pn * (1.0 - np.exp(-s_pn * chl)) - pico_chl
    micro_chl = chl - (cm_pn * (1.0 - np.exp(-s_pn * chl)))
    pico_chl  = np.atleast_1d(np.where(pico_chl>=0,  pico_chl, 0))
    nano_chl  = np.atleast_1d(np.where(nano_chl>=0,  nano_chl, 0))
    micro_chl = np.atleast_1d(np.where(micro_chl>=0, micro_chl, 0))
    if not np.iterable(chl):
        pico_chl  = pico_chl.item()
        nano_chl  = nano_chl.item()
        micro_chl = micro_chl.item()
    return pico_chl, nano_chl, micro_chl
