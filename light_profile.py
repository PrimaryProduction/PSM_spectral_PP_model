
import numpy as np

import config
import scatter
from absorption import calc_ac

def compute_profile(chl_profile,
                    zenith_deg,
                    direct,
                    diffuse,
                    yellow_substance=0.3,
                    wavelengths=None,
                    depths=None):
    depths = config.depths if depths is None else depths
    wavelengths = config.wavelengths if wavelengths is None else wavelengths
    yellow_substance = config.yellow_substance if yellow_substance is None else yellow_substance
    scdict = scatter.all(wavelengths=wavelengths)

    zenw = np.arcsin(np.sin(np.deg2rad(zenith_deg)) / 1.333)

    I0 = direct + diffuse
    mu_d = (direct * np.cos(zenw) + diffuse * 0.831000) / I0
    
    ac,_ = calc_ac(chl_profile, wavelengths)
    ac440idx = np.argmin(np.abs(wavelengths-440))
    ac440 = ac[:,ac440idx]
    ay440 = yellow_substance * ac440
    atot = scdict["aw"] + ac + (ay440[:,np.newaxis] * scdict["ay"]) + 2.0 * scdict["bbr"]


    power = -np.log10(chl_profile)
    bc660 = 0.407 * chl_profile**0.795
    bbtilda = (0.78 + 0.42 * power) * 0.01
    bbtilda[bbtilda < 0.0005] = 0.0005
    bbtilda[bbtilda > 0.01] = 0.01
    
    bc = bc660[:,np.newaxis] * (660.0 / wavelengths)**power[:,np.newaxis]
    bc[bc < 0]=0.0
    bb = bc * bbtilda[:,np.newaxis] + scdict["bw"] * 0.50
    K = (atot + bb) / mu_d
    
    Izlist = [I0,]
    for dz, Ki in zip(np.gradient(depths), K):
        Izlist.append(Izlist[-1] * np.exp(-Ki * dz))
    Iz = np.array(Izlist)[1:,:]
    izpar=(Iz*scdict["d_lambda"]).sum(axis=1)
    euphotic_depth = depths[np.nonzero(izpar/izpar[0]>=0.01)[0].max()]
    return dict(Iz=Iz, IzPAR=izpar, K=K, mu_d=mu_d,
                euphotic_depth=euphotic_depth, atot=atot)

    x=np.log10(100*izpar/izpar[0])[::-1] #need to reverse arrays as np.interp cannot handle decreasing values of x :( 
    y=depth_array[::-1]
    euphotic_depth=np.interp(0,x,y)
    euph_id=np.where(depth_array<euphotic_depth)[0][-1]
    return izpar, Iz, K, euphotic_depth, euph_id, ac