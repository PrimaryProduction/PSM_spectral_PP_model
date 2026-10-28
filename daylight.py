r"""Solar geometry: declination, zenith angle, sunrise and day length.

All functions take a date/time ``dtm`` (anything accepted by
:func:`pandas.to_datetime`) and, where relevant, a latitude in degrees
(positive north). Times are interpreted as **local solar time**: noon is
when the sun crosses the meridian and no longitude or equation-of-time
correction is applied. Longitude therefore does not enter any calculation.

References
----------
.. [1] Spencer, J.W. (1971). Fourier series representation of the position
   of the sun. *Search*, 2(5), 172.
.. [2] Brock, T.D. (1981). Calculating solar radiation for ecological
   studies. *Ecological Modelling*, 14(1-2), 1-19.
   https://doi.org/10.1016/0304-3800(81)90011-9
.. [3] Forsythe, W.C., Rykiel, E.J., Stahl, R.S., Wu, H. & Schoolfield, R.M.
   (1995). A model comparison for daylength as a function of latitude and day
   of year. *Ecological Modelling*, 80(1), 87-95.
   https://doi.org/10.1016/0304-3800(94)00034-F
.. [4] Thekaekara, M.P. & Drummond, A.J. (1971). Standard values for the
   solar constant and its spectral components. *Nature Physical Science*,
   229, 6-9. https://doi.org/10.1038/physci229006a0
.. [5] Iqbal, M. (1983). *An Introduction to Solar Radiation*. Academic Press,
   Toronto. https://doi.org/10.1016/B978-0-12-373750-2.X5001-0
"""

import numpy as np
import pandas as pd

def all(dtm, lat):
    """Compute the solar quantities needed by :func:`surface_light.instant`.

    Parameters
    ----------
    dtm : datetime-like
        Date and local solar time.
    lat : float
        Latitude (degrees north).

    Returns
    -------
    dict
        ``"TOA_Irr"`` : float
            Top-of-atmosphere total solar irradiance (W m-2); see
            :func:`thekaekara`.
        ``"sunrise"`` : pandas.Timestamp
            Time of sunrise on the day of ``dtm``; see :func:`sunrise`.
        ``"daylength"`` : float
            Day length (hours); see :func:`daylength`.
        ``"zenith_rad"`` : float
            Solar zenith angle at ``dtm`` (radians); see :func:`zenith`.

    Examples
    --------
    >>> import daylight
    >>> d = daylight.all("2020-05-06 12:00", 45)
    >>> sorted(d)
    ['TOA_Irr', 'daylength', 'sunrise', 'zenith_rad']
    """
    dtm = pd.to_datetime(dtm)
    sunr, delta, phi = sunrise(dtm, lat, return_all=True)
    return dict(
        TOA_Irr = thekaekara(dtm),
        sunrise = sunr,
        daylength = daylength(dtm, lat),
        zenith_rad = zenith(dtm, lat)
    )

def time_of_day(dtm):
    """Convert a timestamp to decimal hours since midnight.

    Parameters
    ----------
    dtm : datetime-like
        Date and time.

    Returns
    -------
    float or array_like
        Hour of the day as a decimal number (for example 13.5 for 13:30).

    Examples
    --------
    >>> import daylight
    >>> daylight.time_of_day("2020-05-06 13:30")
    13.5
    """
    dtm = pd.to_datetime(dtm)
    #return dtm.hour/24 + dtm.minute/60/24 + dtm.second/60/60/24
    return dtm.hour + dtm.minute/60 + dtm.second/60/60

def thekaekara(dtm):
    """Top-of-atmosphere total solar irradiance for the day of year.

    Linear interpolation in a table of total extraterrestrial irradiance
    against day of year, which accounts for the annual variation of the
    Earth-Sun distance (perihelion in early January, aphelion in early July).

    Parameters
    ----------
    dtm : datetime-like
        Date (only the day of year is used).

    Returns
    -------
    float
        Total solar irradiance at the top of the atmosphere (W m-2).

    Notes
    -----
    The tabulated values (1309-1399 W m-2) correspond to a solar constant of
    1353 W m-2 [1]_ modulated by the Earth-Sun distance. The function is
    ported from the Fortran PSM code; :func:`surface_light.instant` uses it as
    ``TOA_Irr / 1353`` to scale the clear-sky spectra.

    References
    ----------
    .. [1] Thekaekara, M.P. & Drummond, A.J. (1971). Standard values for the
       solar constant and its spectral components. *Nature Physical Science*,
       229, 6-9. https://doi.org/10.1038/physci229006a0

    Examples
    --------
    >>> import daylight
    >>> daylight.thekaekara("2020-01-02")
    1399.0
    >>> round(daylight.thekaekara("2020-06-21"), 1)
    1309.7
    """
    dtm = pd.to_datetime(dtm)
    jday = dtm.day_of_year
    day_points = [
        0., 3., 31., 42., 59., 78., 90., 93., 120., 133., 151., 170.,
        181., 183., 206., 212., 243., 265., 273., 277., 304., 306.,
        334., 355., 365.
    ]
    ir_points = [
        1399., 1399., 1393., 1389., 1378., 1364., 1355., 1353.,
        1332., 1324., 1316., 1310., 1309., 1309., 1312., 1313.,
        1329., 1344., 1350., 1353., 1347., 1375., 1392., 1398., 1399.
    ]

    # Find the first tabulated day on or after jday (as in the Fortran version)
    for i in range(len(day_points)):
        idx = i
        if day_points[i] >= jday:
            break

    if idx == 0:
        return ir_points[idx]
    else:
        temp = (jday - day_points[idx - 1]) / (day_points[idx] - day_points[idx - 1])
        return ir_points[idx - 1] - (ir_points[idx - 1] - ir_points[idx]) * temp

def delta(dtm):
    r"""Compute the solar declination (the delta component of sunrise).

    The declination is the latitude at which the sun is overhead at
    noon.

    Parameters
    ----------
    dtm : datetime-like
        Date (only the day of year is used).

    Returns
    -------
    float or array_like
        Solar declination :math:`\delta` (radians).

    Notes
    -----
    Uses the Fourier series of Spencer (1971) [1]_, accurate to about
    0.0006 rad:

    $$
    \delta = 0.006918 - 0.399912\cos\Gamma + 0.070257\sin\Gamma
             - 0.006758\cos 2\Gamma + 0.000907\sin 2\Gamma
             - 0.002697\cos 3\Gamma + 0.001480\sin 3\Gamma,
    $$

    with the day angle :math:`\Gamma = 2\pi\,d/365` and :math:`d` the day of
    year. (Spencer defines :math:`\Gamma` with :math:`d-1`; the one-day
    offset is retained from the original code.)

    References
    ----------
    .. [1] Spencer, J.W. (1971). Fourier series representation of the
       position of the sun. *Search*, 2(5), 172.

    Examples
    --------
    >>> import numpy as np, daylight
    >>> round(float(np.rad2deg(daylight.delta("2020-06-21"))), 1)
    23.5
    """
    dtm = pd.to_datetime(dtm)
    day_of_year = dtm.day_of_year
    theta = (2 * np.pi * day_of_year) / 365.
    delta = (0.006918 -
             0.399912 * np.cos(theta) +
             0.070257 * np.sin(theta) -
             0.006758 * np.cos(2. * theta) +
             0.000907 * np.sin(2. * theta) -
             0.002697 * np.cos(3. * theta) +
             0.001480 * np.sin(3. * theta))
    return delta



def zenith(dtm, lat):
    r"""Compute the solar zenith angle.

    Parameters
    ----------
    dtm : datetime-like
        Date and local solar time.
    lat : float or array_like
        Latitude (degrees north).

    Returns
    -------
    float or array_like
        Solar zenith angle :math:`\theta_s` (radians). Values greater than
        :math:`\pi/2` mean the sun is below the horizon.

    Notes
    -----
    Standard spherical-astronomy relation (e.g. Iqbal 1983 [1]_):

    $$
    \cos\theta_s = \sin\delta\sin\phi + \cos\delta\cos\phi\cos h,
    \qquad h = \frac{\pi}{12}(t - 12),
    $$

    where :math:`\delta` is the declination (:func:`delta`), :math:`\phi`
    the latitude and :math:`h` the hour angle for local solar time :math:`t`
    in hours.

    References
    ----------
    .. [1] Iqbal, M. (1983). *An Introduction to Solar Radiation*. Academic
       Press, Toronto.

    Examples
    --------
    >>> import numpy as np, daylight
    >>> round(float(np.rad2deg(daylight.zenith("2020-05-06 12:00", 45))), 2)
    28.13
    """
    dtm = pd.to_datetime(dtm)
    tod = time_of_day(dtm)
    dlt = delta(dtm)
    phi = np.deg2rad(lat)
    th  = (tod - 12) * (np.pi / 12)
    zen = np.sin(dlt) * np.sin(phi) + np.cos(dlt) * np.cos(phi) * np.cos(th)
    zen = np.maximum(zen, -1)
    zen = np.minimum(zen,  1)
    return (np.pi / 2.) - np.arcsin(zen)

def zenith_time(dtm, lat, zenith_deg, return_dtm=True):
    r"""Morning time at which the sun reaches a given zenith angle.

    Parameters
    ----------
    dtm : datetime-like
        Date (the time of day is ignored).
    lat : float or array_like
        Latitude (degrees north).
    zenith_deg : float
        Solar zenith angle (degrees).
    return_dtm : bool, default True
        If True return a :class:`pandas.Timestamp` on the day of ``dtm``,
        otherwise decimal hours. Only honoured for scalar input; array input
        always returns decimal hours.

    Returns
    -------
    pandas.Timestamp or float or numpy.ndarray
        Local solar time before noon at which the zenith angle equals
        ``zenith_deg``. Two special values flag the cases where no crossing
        exists: 0 h when the zenith angle stays below ``zenith_deg`` all day
        (e.g. polar day), and -1 h when the sun never gets that high. In the
        latter case a scalar call with ``return_dtm=True`` returns
        ``pandas.NaT``.

    Notes
    -----
    Solves the zenith-angle equation (see :func:`zenith`) for the hour angle,

    $$
    \cos h = \frac{\cos\theta_s - \sin\delta\sin\phi}{\cos\delta\cos\phi},
    $$

    and returns :math:`t = 12 - 12h/\pi`. The model uses
    ``zenith_deg=80`` to start the diurnal integration when the sun is
    sufficiently high for the plane-parallel irradiance model to be valid.

    Examples
    --------
    >>> import daylight
    >>> daylight.zenith_time("2020-05-06", 45, 80)
    Timestamp('2020-05-06 05:49:19')
    >>> daylight.zenith_time("2020-12-21", 80, 80)
    NaT
    """
    dtm = pd.to_datetime(dtm)
    dlt = delta(dtm)
    phi = np.deg2rad(lat)
    zenith_rad  = np.deg2rad(zenith_deg)
    cos_theta   = ((np.cos(zenith_rad) - (np.sin(dlt) * np.sin(phi))) /
                   (np.cos(dlt) * np.cos(phi)))
    with np.errstate(invalid="ignore"):   # |cos_theta| > 1: no crossing
        theta_1 = np.arccos(cos_theta)

    def scalar_theta2time(theta_1):
        if cos_theta <= -1.0:
            local_time = 0.0
        elif cos_theta >= 1.0:
            local_time = -1.0
        else:
            local_time = 12.0 + (theta_1 / (np.pi / 12.0))
        if local_time > 12.0 :
            local_time = 24.0 - local_time
        if return_dtm:
            if local_time < 0:
                return pd.NaT
            return decimal_hour_to_dtm(dtm, local_time)
        return local_time

    def array_theta2time(theta_1):
        local_time = 12.0 + (theta_1 / (np.pi / 12.0))
        local_time[cos_theta <= -1.0] =  0
        local_time[cos_theta >= 1.0]  = -1
        local_time[local_time > 12.0] = 24.0 - local_time[local_time > 12.0]
        return local_time

    try:
        return array_theta2time(theta_1)
    except TypeError:
        return scalar_theta2time(theta_1)

def decimal_hour_to_dtm(dtm, hour):
        """Combine the date of ``dtm`` with a decimal hour.

        Parameters
        ----------
        dtm : pandas.Timestamp
            Timestamp whose date is used (its time of day is discarded).
        hour : float
            Hour of the day as a decimal number.

        Returns
        -------
        pandas.Timestamp
            Midnight of ``dtm`` plus ``hour``, truncated to whole seconds.

        Examples
        --------
        >>> import pandas as pd, daylight
        >>> daylight.decimal_hour_to_dtm(pd.Timestamp("2020-05-06 17:00"), 4.5)
        Timestamp('2020-05-06 04:30:00')
        """
        mn,hr = np.modf(hour)
        sc,mn = np.modf(mn*60)
        _, sc = np.modf(sc*60)
        return dtm.normalize() + pd.Timedelta(f'{hr:02.0f}:{mn:02.0f}:{sc:02.0f}')


def sunrise(dtm, lat, return_dtm=True, return_all=False):
    r"""Compute the time of sunrise from the sunrise hour angle.

    Parameters
    ----------
    dtm : datetime-like
        Date (the time of day is ignored).
    lat : float or array_like
        Latitude (degrees north).
    return_dtm : bool, default True
        If True return a :class:`pandas.Timestamp` on the day of ``dtm``,
        otherwise decimal hours (local solar time).
    return_all : bool, default False
        If True also return the declination and the latitude in radians.

    Returns
    -------
    srise : pandas.Timestamp or float
        Time of sunrise. During polar night sunrise is set to 12:00 and during
        polar day to 00:00.
    delta : float
        Only if ``return_all``. Solar declination (radians), see
        :func:`delta`.
    phi : float
        Only if ``return_all``. Latitude (radians).

    Notes
    -----
    Sunrise occurs at hour angle :math:`h_0` where the zenith angle is
    90 degrees,

    $$
    \cos h_0 = -\tan\phi\tan\delta, \qquad
    t_{\mathrm{rise}} = 12 - \frac{h_0}{15^\circ}\ \mathrm{h}.
    $$

    Atmospheric refraction and the finite size of the solar disc are
    neglected, so this is the *geometric* sunrise.

    Examples
    --------
    >>> import daylight
    >>> daylight.sunrise("2020-05-06", 45)
    Timestamp('2020-05-06 04:49:24')
    """
    dtm = pd.to_datetime(dtm)
    phi = np.deg2rad(lat)
    phidel = -np.tan(phi) * np.tan(delta(dtm))
    phidel = np.maximum(phidel, -1)
    phidel = np.minimum(phidel,  1)
    srise = 12 - np.rad2deg(np.arccos(phidel)) / 15
    if return_dtm:
        srise = decimal_hour_to_dtm(dtm, srise)
    if return_all:
        return srise, delta(dtm), phi
    else:
        return srise

def time_range(dtm, lat, steps=12, zenith_deg=None):
    """Generate a vector of times between sunrise and high noon.

    The times are evenly spaced and symmetric about local noon, running from
    sunrise (or from the morning time when the sun reaches ``zenith_deg``) to
    the corresponding afternoon time.

    Parameters
    ----------
    dtm : datetime-like
        Date (the time of day is ignored).
    lat : float
        Latitude (degrees north).
    steps : int, default 12
        Number of time steps, including both end points.
    zenith_deg : float, optional
        If given, start at the time the sun reaches this zenith angle
        (see :func:`zenith_time`) instead of at sunrise.

    Returns
    -------
    pandas.DatetimeIndex
        ``steps`` evenly spaced times. Empty if the sun never rises (polar
        night) or never reaches ``zenith_deg``, so loops over the result
        simply contribute nothing.

    Examples
    --------
    >>> import daylight
    >>> t = daylight.time_range("2020-05-06", 45, steps=6, zenith_deg=80)
    >>> len(t), t[0].strftime("%H:%M"), t[-1].strftime("%H:%M")
    (6, '05:49', '18:10')
    >>> len(daylight.time_range("2020-12-21", 80, zenith_deg=80))
    0
    """
    dtm = pd.to_datetime(dtm)
    if zenith_deg is None:
        dtm1 = sunrise(dtm, lat)
    else:
        dtm1 = zenith_time(dtm, lat, zenith_deg)
    noon = dtm.normalize() + pd.Timedelta('12:00:00')
    if pd.isna(dtm1) or dtm1 >= noon:
        return pd.DatetimeIndex([])
    return pd.date_range(dtm1, noon+(noon-dtm1), periods=steps)



def daylength(dtm, lat):
    r"""Compute the length of the day using the Brock model.

    The day length is the time between sunrise and sunset, given the day of
    the year and the latitude of the location.

    Parameters
    ----------
    dtm : datetime-like
        Date (only the day of year is used). Day 1 corresponds to
        1 January.
    lat : float or array_like
        Latitude of the location in degrees. Positive values
        for north and negative for south.

    Returns
    -------
    d : float or numpy.ndarray
        Daylength in hours. 24 during polar day and 0 during polar night.

    Notes
    -----
    The Brock (1981) model [1]_ uses the declination approximation of
    Cooper (1969),

    $$
    \delta = 23.45^\circ \sin\left[\frac{360^\circ\,(283 + d)}{365}\right],
    \qquad
    D = \frac{2}{15^\circ}\arccos\left(-\tan\phi\tan\delta\right)\ \mathrm{h},
    $$

    with :math:`d` the day of year. Forsythe et al. (1995) [2]_ compare this
    and other day-length models. Note that this declination formula differs
    slightly from the Spencer series used in :func:`delta`, so ``daylength``
    and ``2 * (12 - sunrise)`` can differ by up to ~17 minutes (within
    60 degrees of the equator).

    References
    ----------
    .. [1] Brock, T.D. (1981). Calculating solar radiation for ecological
       studies. *Ecological Modelling*, 14(1-2), 1-19.
       https://doi.org/10.1016/0304-3800(81)90011-9
    .. [2] Forsythe et al., "A model comparison for daylength as a
       function of latitude and day of year", Ecological Modelling,
       1995, 80(1), 87-95. https://doi.org/10.1016/0304-3800(94)00034-F

    Examples
    --------
    >>> import daylight
    >>> round(daylight.daylength("2020-05-06", 45), 2)
    14.28
    >>> daylight.daylength("2020-06-21", 80)
    24.0
    """
    dtm = pd.to_datetime(dtm)
    day_of_year = dtm.day_of_year
    latInRad = np.deg2rad(lat)
    declinationOfEarth = 23.45*np.sin(np.deg2rad(360.0*(283.0+day_of_year)/365.0))
    mask = -np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth))
    with np.errstate(invalid="ignore"):   # polar day/night, masked below
        hourAngle = np.rad2deg(np.arccos(-np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth))))
    day_hours = np.atleast_1d(2.0*hourAngle/15.0)
    day_hours[mask<=-1.0] = 24.0
    day_hours[mask>= 1.0] = 0
    day_hours = np.squeeze(day_hours)
    return day_hours.item() if day_hours.ndim==0 else day_hours


#def mns_to_doy(mnlist):
#    """_summary_
#
#    Parameters
#    ----------
#    mnlist
#        list of months to convert
#
#    Returns
#    -------
#        _description_
#    """
#    mnlen = [31,28,31,30,31,30,31,31,30,31,30,31]
#    np.cumsum(mnlen) - 15
#    cmnlen = np.cumsum(mnlen) - 15
#    return cmnlen[mnlist-1]
