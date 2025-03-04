
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import KDTree

import config

def interpolate_lut(wavelengths):
    c = np.asarray(
        [
            [1.11, 1.13, 1.18, 1.24, 1.46, 1.70, 2.61],
            [1.04, 1.05, 1.09, 1.11, 1.24, 1.34, 1.72],
            [1.15, 1.00, 1.00, 0.99, 1.06, 1.07, 1.22],
            [1.12, 0.96, 0.96, 0.94, 0.99, 0.96, 1.04],
            [1.32, 1.12, 1.07, 1.02, 1.10, 0.90, 0.80]
        ]
    )
    lut_wl = np.array([400,450,500,550,710])
    intr = interp1d(lut_wl, c, axis=0)
    return intr(wavelengths)

def coefficients(wavelengths=None):
    wavelengths = config.wavelengths if wavelengths is None else wavelengths
    # wavelengths at which to calculate transmittance)
    wls = np.asarray(
            [400., 410., 420., 430., 440., 450., 460., 470.,
             480., 490., 500., 510., 520., 530., 540., 550.,
             570., 593., 610., 630., 656., 667.6, 690., 710.])
    origdict = dict(
        # ozone absorption coefficient
        ao = np.asarray([0.,    0.,    0.,    0.,    0.,    0.003, 0.006, 0.009,
                         0.014, 0.021, 0.030, 0.040, 0.048, 0.063, 0.075, 0.095,
                         0.120, 0.119, 0.132, 0.120, 0.065, 0.060, 0.028, 0.018]),
        # water vapour absorption coefficient
        av = np.asarray([0., 0., 0., 0., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0., 0., 0., 0.,
                         0., 0.075, 0., 0., 0., 0., 0.016, 0.0125]),
        ozone_weights = np.asarray([0., 0., 0., 0., 0., 0., 0., 0.,
                                    0., 0., 0., 0., 0., 0., 0., 0.,
                                    0., 0, 0., 0., 0., 0., 0., 1]),
        dif_weights = np.asarray([0., 0., 0., 0., 0., 0., 0., 0.,
                                    0., 0., 0., 0., 0., 0., 0., 0.,
                                    0., 0, 0., 0., 0., 0., 1., 0]),
        # Extra terrestrial spectral irradiance
        extspir = np.asarray(
            [1479.1, 1701.3, 1740.4, 1587.2, 1837.0, 2005.0, 2043.0, 1987.0,
             2027.0, 1896.0, 1909.0, 1927.0, 1831.0, 1891.0, 1898.0, 1892.0,
             1840.0, 1768.0, 1728.0, 1658.0, 1524.0, 1531.0, 1420.0, 1399.0]),
        )

    outdict =  {key:np.interp(wavelengths, wls, val) for key,val in origdict.items()}
    outdict["wavelengths"] = wavelengths
    outdict["correction_lut"] = interpolate_lut(wavelengths)
    return outdict

cfdict = coefficients()




class Irradiance:

    alpha1 = 1.0274
    beta1  = 0.1324
    alpha2 = 1.206
    beta2  = 0.117
    cd = np.asarray([0., 37., 48.19, 60., 70., 75., 80.])
    _airmass = None
    _air_albedo = None
    
    def __init__(self, zenith_deg, wavelengths=None):
        self.zenith_deg = zenith_deg #np.minimum(zenith_deg, 79.5)
        self.zenith_rad = np.deg2rad(self.zenith_deg)
        if wavelengths is None:
            self.cfdict = cfdict
        else:
            self.cfdict = coefficients(wavelengths)
        self.wld = self.cfdict["wavelengths"] / 1000.

        # Calculate air_albedo with a specified airmass
        self.airmass = 1.9 # Set airmas to a specific number
        self.air_albedo = self.air_albedo # Fix air_albedo
        self.airmass = None # Reset airmass to be calculated


    @property
    def airmass(self):
        if self._airmass is not None :
            return self._airmass
        airmass = 1.0 / (np.cos(self.zenith_rad) + 
                              0.15 * (93.885 - self.zenith_deg) ** (-1.253))
        if hasattr(airmass, "ndim"):
            airmass = np.maximum(airmass, 1)
        else:
            if airmass < 1: airmass = 1
        return airmass
    @airmass.setter
    def airmass(self, value):
        self._airmass = value
    

    @property
    def rayleigh_transmittance(self):
        if hasattr(self.airmass, "ndim"):
            airmass = self.airmass[...,None]
        else:
            airmass = self.airmass
        return np.exp(-airmass / (self.wld**4 * (115.6406 - 1.335 / self.wld**2)))
    @property
    def tr(self):
        return self.rayleigh_transmittance

    @property
    def aerosol_transmittance(self):
        wld = self.wld
        if hasattr(self.tr, "ndim"):
            tr = self.tr[...,None]
        else:
            tr = self.tr
        if hasattr(self.airmass, "ndim"):
            airmass = self.airmass[...,None]
        else:
            airmass = self.airmass
        # AEROSOL SCATTERING AND ABSORBTION
        ta = np.zeros_like(self.tr)
        ta[..., 0:10] = np.exp(-self.beta1 * wld[0:10] ** (-self.alpha1) * airmass)
        ta[..., 10:]  = np.exp(-self.beta2 * wld[10:]  ** (-self.alpha2) * airmass)
        return ta
    @property
    def ta(self):
        return self.aerosol_transmittance

    @property
    def water_vapour_transmittance(self):
        # WATER VAPOUR ABSORPTION
        w = 2.
        # av = water vapour absorption coefficient
        av = self.cfdict["av"]
        if hasattr(self.airmass, "ndim"):
            airmass = self.airmass[...,None]
        else:
            airmass = self.airmass
        return np.exp(-0.3285 * av * (w + (1.42 - w) * 0.5) * 
                         airmass / (1. + 20.07 * av * airmass) ** 0.45)
    @property
    def tw(self):
        return self.water_vapour_transmittance

    @property
    def ozone_transmittance(self):
        airmass = self.airmass
        ao  = self.cfdict["ao"]
        em0 = 35. / ((1224. * (np.cos(self.zenith_rad)) ** 2. + 1.) ** 0.5)
        if hasattr(em0, "ndim"):
            em0 = em0[...,None]
        # old model values
        # self.to = exp(-ao * 0.344 * em0)
        # self.tu = exp(-1.41 * 0.15 * airmass / (1. + 118.3 * 0.15 * airmass) ** 0.45)
        # to be corrected to these values once I have tested equivalency with current codebase
        return np.exp(-ao * 0.344 * em0)
    @property
    def to(self):
        return self.ozone_transmittance
    @property
    def tu(self):
        return np.exp(-1.41 * 0.3 * self.airmass / (1. + 118.3 * 0.3 * self.airmass) ** 0.45)

    @property
    def air_albedo(self):
        if self._air_albedo is not None :
            return self._air_albedo
        ro_S = self.to * self.tw * (self.ta * (1. - self.tr) * 0.5 + self.tr * (1 - self.ta) * 0.22 * 0.928)
        ro_S = (ro_S * (1-self.cfdict["ozone_weights"]) + 
                ro_S * self.tu * self.cfdict["ozone_weights"])
        return ro_S
    @property
    def ro_S(self):
        return self.air_albedo
    @air_albedo.setter
    def air_albedo(self, value):
        self._air_albedo = value


    @property
    def direct_irradiance(self):
        dir_Ir = self.cfdict["extspir"] * self.tr * self.ta * self.tw * self.to
        dir_Ir = (dir_Ir * (1-self.cfdict["ozone_weights"]) + 
                  dir_Ir * self.tu[...,None] * self.cfdict["ozone_weights"])
        return dir_Ir
    @property
    def dir_Ir(self):
        return self.direct_irradiance

    @property
    def diffuse_irradiance(self):
        tree = KDTree(self.cd[:,None])
        zenith_deg = np.atleast_1d(self.zenith_deg)
        dist,ij = tree.query(zenith_deg.flatten()[:,None],2)
        frac = 1 - (dist.T/np.sum(dist, axis=1)).T
        #ij = ij.reshape(zenith_deg.shape)
        corr1 = self.cfdict["correction_lut"][:,ij[:,0].reshape(zenith_deg.shape)]
        corr2 = self.cfdict["correction_lut"][:,ij[:,1].reshape(zenith_deg.shape)]
        c2 = corr1*frac[:,0].reshape(zenith_deg.shape) + corr2*frac[:,1].reshape(zenith_deg.shape)
        c2=c2.T

        extspir = self.cfdict["extspir"]
        zenith_cos = np.cos(self.zenith_rad[...,None])
        xx = extspir * zenith_cos * self.to * self.tw
        r = xx * self.ta * (1. - self.tr) * 0.5
        a = xx * self.tr * (1. - self.ta) * 0.928 * 0.82
        r = (r * (1-self.cfdict["dif_weights"]) + 
             r * self.tu[...,None] * self.cfdict["dif_weights"])
        a = (a * (1-self.cfdict["dif_weights"]) + 
             a * self.tu[...,None] * self.cfdict["dif_weights"])
        g = (self.dir_Ir*zenith_cos + (r + a) * c2) * self.ro_S*0.05 / (1. - 0.05*self.ro_S)
        return np.squeeze((r + a) * c2 + g)
    @property
    def dif_Ir(self):
        return self.diffuse_irradiance

    @property
    def irradiance(self):
        return {'direct':  self.direct_irradiance,

                'diffuse': self.diffuse_irradiance}

def all(zenith_deg, wavelengths=None):
    ir = Irradiance(zenith_deg, wavelengths=wavelengths)
    return ir.irradiance

