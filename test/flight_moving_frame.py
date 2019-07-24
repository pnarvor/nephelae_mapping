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

def get_coordinate_extent(atm):

    tvar = atm.variables['time'][:]
    tvar = tvar - tvar[0]
    xvar = 1000.0 * atm.variables['W_E_direction'][:]
    yvar = 1000.0 * atm.variables['S_N_direction'][:]
    zvar = 1000.0 * np.squeeze(atm.variables['VLEV'][:, 0, 0])
    return tvar, zvar, xvar, yvar

def show_map(data, xy_extent, data_unit, data_extent, time_stamp, height, figsize=(8, 6)):

    fig = plt.figure(figsize=figsize)
    plt.title("Cloud at t= %ds & z= %dm"%(time_stamp, height))
    plt.xlabel("x coordinate")
    plt.ylabel("y coordinate")
    plot = plt.imshow(data.T, origin='lower', extent=xy_extent, cmap=plt.cm.viridis)

    cbar = plt.colorbar(fraction=0.0471, pad=0.01)
    cbar.ax.set_title(data_unit, pad=14)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    plt.clim(data_extent[0], data_extent[1])
    plt.tight_layout()
    return fig, plot

def line_trajectory(xi, yi, t, theta=0, s=20):

    x_new = xi + s * np.cos(np.radians(theta)) * t
    y_new = yi + s * np.sin(np.radians(theta)) * t
    return x_new, y_new

def circ_trajectory(xc, yc, t, r, v=1000):

    w = v/r
    x_new = xc + r * np.cos(np.radians(w * t))
    y_new = yc + r * np.sin(np.radians(w * t))
    return x_new, y_new

def lemniscate_trajectory(xc, yc, t, r, v=1000):

    alpha = r
    w = v / r
    x_new = xc + alpha * np.sqrt(2) * np.cos(np.radians(90+ (w * t))) / (np.sin(np.radians(90+ (w * t))) ** 2 + 1)
    y_new = yc + alpha * np.sqrt(2) * np.cos(np.radians(90+ (w * t))) * np.sin(np.radians(90+ (w * t))) / (np.sin(np.radians(90+ (w * t))) ** 2 + 1)
    return x_new, y_new

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point
    qx = ox + m.cos(angle) * (px - ox) - m.sin(angle) * (py - oy)
    qy = oy + m.sin(angle) * (px - ox) + m.cos(angle) * (py - oy)
    return qx, qy


if __name__ == "__main__":

    gt_save = False
    anim_save = False
    save_path = "exp/3/"
    if(anim_save):
        matplotlib.use("Agg")
        plt.rcParams['animation.ffmpeg_path'] = '/home/dlohani/miniconda3/envs/nephelae/bin/ffmpeg'

    atm = MFDataset("/net/skyscanner/volume1/data/Nephelae/MesoNH-2019-02/REFHR.1.ARMCu.4D.nc")
    tvar, zvar, xvar, yvar = get_coordinate_extent(atm)

    var_interest = 'RCT' # or 'WT' etc
    data = MesoNHVariable(atm, var_interest, interpolation='linear')
    lwc_unit = "kg a/kg w"

    tStart = 180  # initial timestamp
    time_length = 215 # length of time span
    z = 1075.0    # fixed height in m
    xSlice = slice(1500, 3800, None)
    ySlice = slice(4100, 4800, None)
    lwc_extent = [data[:,z,xSlice,ySlice].data.min(), data[:,z,xSlice,ySlice].data.max() + 0.25 * data[:,z,xSlice,ySlice].data.max()]

    xyBounds = data[0.0, z, xSlice, ySlice].bounds
    xyExtent = [xyBounds[0][0], xyBounds[0][1], xyBounds[1][0], xyBounds[1][1]]

    fig, lwc_data = show_map(np.zeros(data[0.0, z, ySlice, xSlice].data.shape), xyExtent, lwc_unit, lwc_extent, 0, z, (12,4))

    traj_type = "lemniscate" # or circular
    # Trajectory params
    xi = 1700           # intial drone position
    yi = 4400           # intial drone position
    s_w = 8             # wind speed
    theta = 0           # angle(speed vector, x axis)
    radius = 80         # r of circle or focus of lemniscate
    v_circle = 400      # speed in curve
    end_reach = False

    # For lemniscate trajectories
    rot_active = False
    angles = [90, 45, 135]
    colors = ["red", "green", "blue", "yellow", "black"]
    color = "white"
    rot_angle = 0
    sub_factor = 0
    xf = 0
    yf = 0

    XT = np.empty((0, 3), float)
    yT = []

    def init():
        #do nothing
        pass

    def update(i):

        t = tStart + i

        lwc_data.set_data(data[t, z, ySlice, xSlice].data)
        plt.title("Drone following cloud characterized by LWC at t= %ds" % (t))

        global end_reach, rot_active, sub_factor, angles, rot_angle, xf, yf, colors, color, XT, yT

        if (end_reach == False and i < time_length-1):

            if(traj_type == "lemniscate"):
                xn, yn = lemniscate_trajectory(xi, yi, i - sub_factor, radius, v_circle)
                if (xn == xf and yn == yf):
                    rot_active = True
                    sub_factor = i - 1
                    if (angles):
                        rot_angle = angles.pop(0)
                        color = colors.pop(0)
                    else:
                        rot_active = False
                        angles = [90, 45, 135]
                        rot_angle = 0
                        color = "white"
                        colors = ["red", "green", "blue", "yellow", "black"]
                if (i == 1):
                    xf, yf = xn, yn
                if (rot_active):
                    xn, yn = rotate((xi, yi), (xn, yn), m.radians(rot_angle))

            elif(traj_type == "circular"):
                xn, yn = circ_trajectory(xi, yi, i , radius, v_circle)

            else:
                print("Wrong trajectory type! Exiting...")
                exit()

            xn, yn = line_trajectory(xn, yn, i, theta, s_w)  # adding wind speed

            if (xn >= xyExtent[1] - 200 or yn >= xyExtent[3] - 200):
                end_reach = True
            else:
                XT = np.vstack([XT, [xn, yn, t]])
                yT = np.append(yT, data[t, z, yn, xn])
                plt.scatter(xn, yn, c=color, marker="x", s=15)

        if (i == time_length-1 and gt_save):

            # Save 2D area info + var_interest limits
            xExtent = xvar[(xvar >= xyExtent[0]) & (xvar <= xyExtent[1])]
            yExtent = yvar[(yvar >= xyExtent[2]) & (yvar <= xyExtent[3])]
            save_pickle(xExtent, save_path+"xExtent")
            save_pickle(yExtent, save_path+"yExtent")
            save_pickle(lwc_extent, save_path+"lwc_extent")

            # Save Trajectory and data info
            save_pickle(XT, save_path+"coord")
            save_pickle(yT.reshape(-1, 1), save_path+"lwc_data")

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=range(1, time_length),
        init_func=init,
        repeat=False,
        interval=1,
    )

    if (anim_save):
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
        anim.save(save_path+'gt_flight.mp4', writer=writer)
    else:
        plt.show(block=False)