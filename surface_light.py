
import numpy as np
import pandas as pd

import absorption
import irradiance
import daylight
import scatter
import config

import irradiance as bird_irr

def instant(dtm, lat, par, wavelengths=None, cloud=0, verbose=False):
    """_summary_

    Parameters
    ----------
    dtm
        _description_
    lat
        _description_
    par
        _description_
    wavelengths, optional
        _description_, by default None
    cloud, optional
        _description_, by default 0
    verbose, optional
        _description_, by default False
    """
    wavelengths = config.wavelengths if wavelengths is None else wavelengths
    scdict = scatter.all(wavelengths=wavelengths)
 
    #Calculate daylengths etc
    daydict = daylight.all(dtm, lat)
    zenithR = daydict["zenith_rad"]
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
    i_direct  =  np.sum(direct * np.cos(zenithR))
    i_diffuse =  np.sum(diffuse)
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
    ref = 0.5 * (np.sin(zenithR - zenithW)) ** 2 / (np.sin(zenithR + zenithW)) ** 2
    ref += 0.5 * (np.tan(zenithR - zenithW)) ** 2 / (np.tan(zenithR + zenithW)) ** 2

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
    frac_of_day = ((dtm - daydict["sunrise"]).seconds/3600)/daydict["daylength"]
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

