
import numpy as np

from config import phyto_wl

cm_pn = 0.77
s_pn = 0.94 / cm_pn
cm_p = 0.13
s_p = 0.80 / cm_p

def brewin(chl):
    '''
    Calculate Chl size fractions for different phytoplankton size classes.
    
    Ref
    ---
    DOI:10.1016/j.ecolmodel.2010.02.014
    DOI:10.1002/2014JC009859
    '''
    #compute fractions
    pico_chl  = cm_p * (1.0 - np.exp(-s_p * chl))
    nano_chl  = cm_pn * (1.0 - np.exp(-s_pn * chl)) - pico_chl
    micro_chl = chl - (cm_pn * (1.0 - np.exp(-s_pn * chl)))
    pico_chl  = np.atleast_1d(np.where(pico_chl>=0,  pico_chl, 0))
    nano_chl  = np.atleast_1d(np.where(nano_chl>=0,  nano_chl, 0))
    micro_chl = np.atleast_1d(np.where(micro_chl>=0, micro_chl, 0))
    if not np.iterable(chl):
        pico_chl  = pico_chl.item()
        nano_chl  = nano_chl.item()
        micro_chl = micro_chl.item()
    return pico_chl, nano_chl, micro_chl
