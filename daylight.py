
import numpy as np
import pandas as pd

def all(dtm, lat):
    dtm = pd.to_datetime(dtm)
    sunr, delta, phi = sunrise(dtm, lat, return_all=True)
    return dict(
        TOA_Irr = thekaekara(dtm),
        sunrise = sunr,
        daylength = daylength(dtm, lat),
        zenith_rad = zenith(dtm, lat)
    )

def time_of_day(dtm):
    dtm = pd.to_datetime(dtm)
    #return dtm.hour/24 + dtm.minute/60/24 + dtm.second/60/60/24
    return dtm.hour + dtm.minute/60 + dtm.second/60/60

def thekaekara(dtm):
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

    # NOTE (jad): I would personally use np.argmin here but I want the initial version to closely follow
    # what was done in the fortran version
    for i in range(len(day_points)):
        idx = i
        if i >= jday:
            break

    if idx == 0:
        return ir_points[idx]
    else:
        temp = (jday - day_points[idx - 1]) / (day_points[idx] - day_points[idx - 1])
        return ir_points[idx - 1] - (ir_points[idx - 1] - ir_points[idx]) * temp

def delta(dtm):
    """Compute delta component of sunrise"""
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
    """Compute Zenith using XXX
    
    Ref
    ---
    YYY
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
    dtm = pd.to_datetime(dtm)
    dlt = delta(dtm)
    phi = np.deg2rad(lat)
    zenith_rad  = np.deg2rad(zenith_deg)
    cos_theta   = ((np.cos(zenith_rad) - (np.sin(dlt) * np.sin(phi))) / 
                   (np.cos(dlt) * np.cos(phi)))
    theta_1     = np.arccos(cos_theta)

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
        mn,hr = np.modf(hour)
        sc,mn = np.modf(mn*60)
        _, sc = np.modf(sc*60)
        return dtm.normalize() + pd.Timedelta(f'{hr:02.0f}:{mn:02.0f}:{sc:02.0f}')


def sunrise(dtm, lat, return_dtm=True, return_all=False):
    """Compute the XXX sunrise using YYY
    
    
    Ref
    ---

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
        return srise, delta, phi
    else:
        return srise

def time_range(dtm, lat, steps=12, zenith_deg=None):
    """Generate a vector of times between sunrise and high noon."""
    dtm = pd.to_datetime(dtm)
    if zenith_deg is None:
        dtm1 = sunrise(dtm, lat)
    else:
        dtm1 = zenith_time(dtm, lat, zenith_deg)
    noon = dtm.normalize() + pd.Timedelta('12:00:00')
    return pd.date_range(dtm1, noon+(noon-dtm1), periods=steps)



def daylength(dtm, lat):
    """Computes the length of the day (the time between sunrise and
    sunset) given the day of the year and latitude of the location.
    Function uses the Brock model for the computations.
   
    Parameters
    ----------
    dayOfYear : int
        The day of the year. 1 corresponds to 1st of January
        and 365 to 31st December (on a non-leap year).
    lat : float
        Latitude of the location in degrees. Positive values
        for north and negative for south.
    
    Returns
    -------
    d : float
        Daylength in hours.
   
    Ref
    ---
    Forsythe et al., "A model comparison for daylength as a
    function of latitude and day of year", Ecological Modelling,
    1995. 
      
    """
    dtm = pd.to_datetime(dtm)
    day_of_year = dtm.day_of_year
    latInRad = np.deg2rad(lat)
    declinationOfEarth = 23.45*np.sin(np.deg2rad(360.0*(283.0+day_of_year)/365.0))
    mask = -np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth))
    hourAngle = np.rad2deg(np.arccos(-np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth))))
    day_hours = np.atleast_1d(2.0*hourAngle/15.0)
    day_hours[mask<=-1.0] = 24.0
    day_hours[mask>= 1.0] = 0
    day_hours = np.squeeze(day_hours)
    return day_hours.item() if day_hours.ndim==0 else day_hours
    #return 2 * (12. - sunrise(dtm, lat, return_dtm=False))


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


####
    sunrise, delta, phi = computeSunrise(iday, lat)
    zen_80_time = compute_zenith_time(delta, phi, 80)
    times = genTimeArray(zen_80_time, num_time_steps)
    zenR_array = np.asarray([computeZenith(x, delta, phi) for x in times])
    delta_t = (12.0 - sunrise) / len(times)
    start_time_idx = 0
    start_time = zen_80_time
    daylength = 2 * (12.0 - sunrise)
    iom = par * np.pi / (2 * daylength)
    delta_prestart = start_time - sunrise
    if debug:
        print('Daylength: ', daylength)
        print('iom: ', iom)
