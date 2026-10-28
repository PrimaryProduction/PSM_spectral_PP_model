"""Shared model configuration: spectral grid, depth grid and default constants.

Every module in the model imports its default wavelength and depth grids from
here, so changing a value in this module changes the default resolution of the
whole model.

Attributes
----------
phyto_wl : numpy.ndarray
    Wavelengths (nm) at which the tabulated optical properties are defined.
depths : numpy.ndarray
    Default vertical grid (m).
yellow_substance : float
    Default ratio of CDOM absorption to phytoplankton absorption at 440 nm.
wavelengths : numpy.ndarray
    Default wavelength grid used in the model calculations (nm).

Notes
-----
The tabulated spectra in :mod:`absorption` (phytoplankton size-class specific
absorption) and :mod:`scatter` (pure-water absorption) are given on
:data:`phyto_wl`, so that grid is the native resolution of the model. Any
other grid passed through the ``wavelengths`` keyword of the model functions
is linearly interpolated from these tables.
"""

import numpy as np

#from scatter import calc_aw, calc_ay, calc_bbr, calc_bw

phyto_wl = np.array([400, 405, 410, 415, 420, 425, 430, 435, 440, 445, 450, 455,
                     460, 465, 470, 475, 480, 485, 490, 495, 500, 505, 510, 515,
                     520, 525, 530, 535, 540, 545, 550, 555, 560, 565, 570, 575,
                     580, 585, 590, 595, 600, 605, 610, 615, 620, 625, 630, 635,
                     640, 645, 650, 655, 660, 665, 670, 675, 680, 685, 690, 695,
                     700])
"""Wavelengths (nm) of the tabulated spectra: 400-700 nm in 5 nm steps (61 bands)."""

depths = np.arange(0,250,0.5)
"""Default depth grid (m): 0-249.5 m at 0.5 m resolution (500 levels)."""

yellow_substance = 0.3
"""Default ratio a_y(440)/a_ph(440) of CDOM (yellow substance) to phytoplankton absorption."""

wavelengths = phyto_wl
"""Default wavelength grid (nm) used by the model; identical to :data:`phyto_wl`."""

"""def attenuation_dataframe(wavelengths=None):

    wls = phyto_wl if wavelengths is None else wavelengths
    ay  = calc_ay(wls)
    bbr = calc_bbr(wls)
    bw  = calc_bw(wls)
    aw  = calc_aw(wls)
    d_lambda =  np.append(np.diff(wls), np.diff(wls)[-1])
    return pd.DataFrame({"wls":wls, "ay":ay, "bbr":bbr, "bw":bw, "aw":aw,
                         "d_lambda":d_lambda})"""

"""test_case = dict(year = 2010,
month = 6
date = dt.date(year, month, 1)
date = date + dt.timedelta(days=14)
iday = date.timetuple().tm_yday
lat=60
lon=-30
chl = 1.5
par = 25.0
mld = 30.
sigma=24.0
rho = 0.86
zm = 60
bathymetry = 200
b_zero = chl / (1 + (rho / (1 - rho)) * np.exp(-zm ** 2 / (2 * sigma ** 2)))
h =  sigma * (rho / (1 - rho)) * b_zero * np.sqrt(2 * np.pi)
Cloud=0.0
yellow_substance=0.3
num_time_steps=12}"""
