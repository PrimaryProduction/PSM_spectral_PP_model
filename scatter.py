
import numpy as np

import config

"""
Scattering Coefficients pure seawater at 500nm
"""

bw500 = 0.00288  # Scattering coefficient of pure sea-water at 500nm
br488 = 0.00027

def all(wavelengths=None):
    wavelengths = config.wavelengths if wavelengths is None else wavelengths
    return dict(
        ay = calc_ay(wavelengths),
        bbr = calc_bbr(wavelengths),
        bw = calc_bw(wavelengths),  
        aw = calc_aw(wavelengths),
        d_lambda = np.append(np.diff(wavelengths), np.diff(wavelengths)[-1]))

def calc_bw(wavelengths=None):
    """Pure sea-water scattering"""
    wavelengths = config.wavelengths if wavelengths is None else wavelengths
    bw = bw500 * (wavelengths / 500.) ** -4.3
    return bw

def calc_bbr(wavelengths=None):
    """Raman scattering in pure sea-water
    
    Ref
    ===
    Jasmine Bartlett
    """ 
    wavelengths = config.wavelengths if wavelengths is None else wavelengths
    bbr = 0.5 * br488 * (wavelengths/ 488.) ** -5.3
    return bbr

def calc_ay(wavelengths=None):
    """Absorption by CDOM"""
    wavelengths = config.wavelengths if wavelengths is None else wavelengths
    ay = np.exp(-0.014 * (wavelengths - 440))
    return ay

def calc_aw(wavelengths=None):
    """Absorption spectra of pure water

    REF
    ---
    DOI:10.1364/AO.36.008710 (Pope & Fry 1997)
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

