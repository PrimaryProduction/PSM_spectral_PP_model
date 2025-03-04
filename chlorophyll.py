
import numpy as np

import config

def sinusodal_profile(chl, sigma=24, rho=0.86, z_m=60, depths=None, mld=None):
    """Generate a sinusodial Chlorophyll depth profile 
    
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
