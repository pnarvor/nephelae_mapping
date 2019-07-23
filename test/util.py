################################################################
############ Utility functions #######
################################################################

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
import matplotlib.colors as mcolors
import pickle
import sys
from netCDF4 import MFDataset
from nephelae_simulation.mesonh_interface import MesoNHVariable
from nephelae_mapping.test.mesonh_atm.mesonh_atmosphere import MesoNHAtmosphere

def save_pickle(data, name):

    pickle_out = open( name+".pickle", "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()

def load_pickle(name):

    pickle_in = open(name+".pickle", "rb")
    data = pickle.load(pickle_in) #, encoding='latin1')
    pickle_in.close()
    return data