

import numpy as np
import pandas as pd

#from scatter import calc_aw, calc_ay, calc_bbr, calc_bw

phyto_wl = np.array([400, 405, 410, 415, 420, 425, 430, 435, 440, 445, 450, 455, 
                     460, 465, 470, 475, 480, 485, 490, 495, 500, 505, 510, 515,
                     520, 525, 530, 535, 540, 545, 550, 555, 560, 565, 570, 575,
                     580, 585, 590, 595, 600, 605, 610, 615, 620, 625, 630, 635, 
                     640, 645, 650, 655, 660, 665, 670, 675, 680, 685, 690, 695, 
                     700])
depths = np.arange(0,250,0.5)
yellow_substance = 0.3

wavelengths = phyto_wl

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