import numpy as np

from config import phyto_wl

from size_fraction import brewin as calc_psd

"""
Specific absorption for different phyto plankton types
"""
NANO  = np.asarray([0.01600000, 0.017600,   0.02426667, 0.0265625, 0.02776562,  0.02896875, 
                    0.03017188, 0.031375,   0.03257812, 0.03241667, 0.03020833, 0.0280000,
                    0.03135000, 0.034700,   0.03242917, 0.03015833, 0.0278875,  0.02561667,
                    0.02326190, 0.02057143, 0.01788095, 0.01519048, 0.0125,     0.0114, 
                    0.0103,     0.0093,     0.0083,     0.007125,   0.00595,    0.004775, 
                    0.0036,     0.0024,     0.0017,     0.0007,     0.0005,     0.001325,
                    0.00215,    0.002975,   0.0038,     0.00390345, 0.0040069,  0.00411034,
                    0.00421379, 0.00431724, 0.00443333, 0.0046,     0.0057375,  0.006875,
                    0.0080125,  0.00915,    0.0102875,  0.011425,   0.0125625,  0.0137,
                    0.0199,     0.02020769, 0.02051538, 0.01826471, 0.01217647, 0.00608824, 0.])
PICO  = np.asarray([0.09572727, 0.1053,     0.11313333, 0.1197625,  0.12609063, 0.13241875,
                    0.13874687, 0.145075,   0.15140312, 0.15406667, 0.15123333, 0.1484, 
                    0.1401,     0.1318,     0.12582083, 0.11984167, 0.1138625,  0.10788333,
                    0.1011,     0.0911,     0.0811,     0.0711,     0.0611,     0.0514, 
                    0.0417,     0.0351,     0.0285,     0.025625,   0.02275,    0.019875,
                    0.017,      0.0155,     0.0136,     0.0131,     0.0122,     0.011525,
                    0.01085,    0.010175,   0.0095,     0.00982759, 0.01015517, 0.01048276,
                    0.01081034, 0.01113793, 0.01166667, 0.013,      0.014925,   0.01685,
                    0.018775,   0.0207,     0.022625,   0.02455,    0.026475,   0.0284,
                    0.0348,     0.03156923, 0.02833846, 0.02329412, 0.01552941, 0.00776471, 0])
MICRO = np.asarray([0.01518182, 0.0167,     0.01686667, 0.0176,     0.018475,   0.01935,
                    0.020225,   0.0211,     0.021975,   0.02213333, 0.02121667, 0.0203,
                    0.01925,    0.0182,     0.0173875,  0.016575,   0.0157625,  0.01495,
                    0.0141381,  0.01332857, 0.01251905, 0.01170952, 0.0109,     0.0101,
                    0.0093,     0.0087,     0.0081,     0.0075,     0.0069,     0.0063,
                    0.0057,     0.0049,     0.0043,     0.0038,     0.0037,     0.0038,
                    0.0039,     0.004,      0.0041,     0.00428966, 0.00447931, 0.00466897,
                    0.00485862, 0.00504828, 0.00526667, 0.0056,     0.0066125,  0.007625,
                    0.0086375,  0.00965,    0.0106625,  0.011675,   0.0126875,  0.0137,
                    0.0165,     0.01480769, 0.01311538, 0.01067647, 0.00711765, 0.00355882, 0.])



def calc_ac(chl, wavelengths=None, size_class="all"):
    """Calculate absorption by different phytoplankton size classes.
    
    
    coefficients taken from Brewin et al 2011 and 2015"""
    pico_chl, nano_chl, micro_chl = calc_psd(chl)

    if wavelengths is not None:
        pico_abs  = np.interp(wavelengths, phyto_wl, PICO)
        nano_abs  = np.interp(wavelengths, phyto_wl, NANO)
        micro_abs = np.interp(wavelengths, phyto_wl, MICRO)
    else:
        pico_abs  = PICO
        nano_abs = NANO
        micro_abs = MICRO

    if np.iterable(chl):
        absorption = ((pico_abs  * pico_chl[...,None]) + 
                      (nano_abs  * nano_chl[...,None]) + 
                      (micro_abs * micro_chl[...,None]))
    else:
        if "n" in size_class.lower():
            print("nano")
            absorption = nano_abs*nano_chl 
        elif "m" in size_class.lower():
            absorption = micro_abs*micro_chl 
        elif "p" in size_class.lower():
            absorption = pico_abs*pico_chl 
        else:
            absorption = (pico_abs*pico_chl + nano_abs*nano_chl + micro_abs*micro_chl)

    mean_absorption = absorption.mean(axis=-1)
    return (absorption, np.squeeze(mean_absorption))


