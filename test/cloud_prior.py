#! /usr/bin/python3

import os
import time
import numpy as np
import math as m
import matplotlib.pyplot as plt
from matplotlib import animation
import pickle
import matplotlib
from netCDF4 import MFDataset
from nephelae_simulation.mesonh_interface import ScaledArray
from nephelae_simulation.mesonh_interface import DimensionHelper
from nephelae_simulation.mesonh_interface import MesoNHVariable
from nephelae_mapping.test.util import save_pickle
from nephelae_mapping.test.process_map import border_cs
from nephelae_mapping.test.flight_moving_frame import get_coordinate_extent, show_map


if __name__ == "__main__":

    gt_save = False
    anim_save = False
    save_path = "exp/4/"
    if(anim_save):
        matplotlib.use("Agg")
        plt.rcParams['animation.ffmpeg_path'] = '/home/dlohani/miniconda3/envs/nephelae/bin/ffmpeg'

    atm = MFDataset("/net/skyscanner/volume1/data/Nephelae/MesoNH-2019-02/REFHR.1.ARMCu.4D.nc")
    tvar, zvar, xvar, yvar = get_coordinate_extent(atm)

    var_interest = 'RCT' # or 'WT' etc
    data = MesoNHVariable(atm, var_interest, interpolation='linear')
    lwc_unit = "kg a/kg w"

    t = 211        # inital timestamp
    zStart = 1002  # initial height  ini m
    z_depth = 128  # length of z to travel in cloud
    xSlice = slice(1670, 2200, None)
    ySlice = slice(4170, 4700, None)
    lwc_extent = [data[t,:,xSlice,ySlice].data.min(), data[t,:,xSlice,ySlice].data.max() + 0.25 * data[t,:,xSlice,ySlice].data.max()]

    xyBounds = data[t, zStart, xSlice, ySlice].bounds
    xyExtent = [xyBounds[0][0], xyBounds[0][1], xyBounds[1][0], xyBounds[1][1]]

    data_shape = data[t, zStart, ySlice, xSlice].data.shape
    fig, lwc_data = show_map(np.zeros(data_shape), xyExtent, lwc_unit, lwc_extent, 0, 0)
    _, border_data = border_cs(np.zeros((data_shape[0]*data_shape[1],1)), data_shape, xyExtent, threshold=3e-5, c="Black")

    def init():
        #do nothing
        pass

    def update(i):

        z = zStart + i
        global  border_data
        lwc_data_z = data[t, z, ySlice, xSlice].data
        lwc_data.set_data(lwc_data_z)
        for coll in border_data.collections:
            plt.gca().collections.remove(coll)
        _, border_data = border_cs(lwc_data_z.T, data_shape, xyExtent, threshold=3e-5, c="Black")
        plt.title("Cloud at t= %ds & z= %dm" % (t, z))

    anim = animation.FuncAnimation(
        fig,
        update,
        frames = z_depth,
        init_func = init,
        repeat = False,
        interval = 200,
    )

    if (anim_save):
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
        anim.save(save_path+'gt.mp4', writer=writer)
    else:
        plt.show(block=False)