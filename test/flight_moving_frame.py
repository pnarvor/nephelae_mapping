#! /usr/bin/python3

import os
import time
import numpy as np
import math as m
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib
# matplotlib.use("Agg")
# plt.rcParams['animation.ffmpeg_path'] = '/home/dlohani/miniconda3/envs/nephelae/bin/ffmpeg'

from netCDF4 import MFDataset

from nephelae_simulation.mesonh_interface import ScaledArray
from nephelae_simulation.mesonh_interface import DimensionHelper
from nephelae_simulation.mesonh_interface import MesoNHVariable

var0 = 'RCT'
# var1 = 'UT'     # WE wind
# var1 = 'VT'     # SN wind
var1 = 'WT'  # vertical wind

atm = MFDataset('/net/skyscanner/volume1/data/Nephelae/MesoNH-2019-02/REFHR.1.ARMCu.4D.nc')

tvar = atm.variables['time'][:]
tvar = tvar - tvar[0]
xvar = 1000.0 * atm.variables['W_E_direction'][:]
yvar = 1000.0 * atm.variables['S_N_direction'][:]
zvar = 1000.0 * np.squeeze(atm.variables['VLEV'][:, 0, 0])

# data0 = MesoNHVariable(atm, var0, interpolation='nearest')
# data1 = MesoNHVariable(atm, var1, interpolation='nearest')
data0 = MesoNHVariable(atm, var0, interpolation='linear')
#data1 = MesoNHVariable(atm, var1, interpolation='linear')


z0 = 1075.0
y0 = 4500.0
xySlice = slice(xvar[0]+0.5, xvar[-1]-0.5, None)
#zSlice = slice(0.0, 12000.0, None)
tStart = 70 # initial timestamp

xyBounds = data0[0.0, z0, xySlice, xySlice].bounds
xyExtent = [xyBounds[0][0], xyBounds[0][1], xyBounds[1][0], xyBounds[1][1]]
# xzBounds = data0[0.0, zSlice, xySlice, y0].bounds
# xzExtent = [xzBounds[1][0], xzBounds[1][1], xzBounds[0][0], xzBounds[0][1]]

print('Started !')

fig = plt.figure()
varDisp0 = plt.imshow(data0[tStart, z0, xySlice, xySlice].data, cmap=plt.cm.viridis, origin='lower', extent=xyExtent)
# fig, axes = plt.subplots(2, 2, sharex=True)
# varDisp0 = axes[0][0].imshow(data0[0.0, z0, xySlice, xySlice].data, cmap=plt.cm.viridis, origin='lower',
#                              extent=xyExtent)
#varDisp0 = plt.imshow(data0[0.0, z0, xySlice, xySlice].data, cmap=plt.cm.viridis, origin='lower', extent=xyExtent)
# varDisp1 = axes[1][0].imshow(data0[0.0, zSlice, y0, xySlice].data, cmap=plt.cm.viridis, origin='lower', extent=xzExtent)
# varDisp2 = axes[0][1].imshow(data1[0.0, z0, xySlice, xySlice].data, cmap=plt.cm.viridis, origin='lower',
#                              extent=xyExtent)
# varDisp3 = axes[1][1].imshow(data1[0.0, zSlice, y0, xySlice].data, cmap=plt.cm.viridis, origin='lower', extent=xzExtent)

xi = 400
yi = 2100
s_w = 8 # wind speed
theta = 0 # angle(speed vector, x axis)
radius = 100
v_circle = 400 # speed in circle
end_reach = False

# for lemniscate trajectories
rot_active = False
angles = [90, 45, 135]
colors = ["red", "green", "blue", "yellow", "black"]
color = "white"
rot_angle = 0
sub_factor = 0
xf = 0
yf = 0

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

def init():
    # axes[0][0].scatter(xi, yi, c="white", marker="x", s=20)
    plt.plot([xySlice.start, xySlice.stop], [y0, y0], color=[0.0, 0.0, 0.0, 1.0])
    #plt.scatter(xi, yi, c="white", marker="x", s=15)
    # axes[0][0].plot([xySlice.start, xySlice.stop], [y0, y0], color=[0.0, 0.0, 0.0, 1.0])
    # axes[1][0].plot([zSlice.start, zSlice.stop], [z0, z0], color=[0.0, 0.0, 0.0, 1.0])
    # axes[0][1].plot([xySlice.start, xySlice.stop], [y0, y0], color=[0.0, 0.0, 0.0, 1.0])
    # axes[1][1].plot([zSlice.start, zSlice.stop], [z0, z0], color=[0.0, 0.0, 0.0, 1.0])


def update(i):

    t = tStart + i

    varDisp0.set_data(data0[t, z0, xySlice, xySlice].data)
    # varDisp1.set_data(data0[t, zSlice, y0, xySlice].data)
    # varDisp2.set_data(data1[t, z0, xySlice, xySlice].data)
    # varDisp3.set_data(data1[t, zSlice, y0, xySlice].data)

    global end_reach, rot_active, sub_factor, angles, rot_angle, xf, yf, colors, color

    if (end_reach == False):

        xn, yn = lemniscate_trajectory(xi, yi, i - sub_factor, radius, v_circle)
        xc, yc = xn, yn
        print(t, sub_factor, i-sub_factor, xf, yf, xc, yc)
        if (xc==xf and yc==yf):
            rot_active = True
            sub_factor = i - 1
            print(angles, colors)
            if (angles):
                rot_angle = angles.pop(0)
                color =colors.pop(0)
            else:
                rot_active = False
                angles = [90, 45, 135]
                rot_angle = 0
                color = "white"
                colors = ["red", "green", "blue", "yellow", "black"]

        if (rot_active):
            xn, yn = rotate((xi, yi), (xn, yn), m.radians(rot_angle))
        if(i==1):
            xf, yf = xc, yc
        #print(xi, yi, i - sub_factor, xc, yc, rot_active)
        # Circular + wind
        #xn, yn = circ_trajectory(xi, yi, i , radius, v_circle)
        xn, yn = line_trajectory(xn, yn, i , theta, s_w) # adding wind speed

        if (xn >= xyExtent[1] - 200 or yn >= xyExtent[3] - 200):
            end_reach = True
        else:
           plt.scatter(xn, yn, c=color, marker="x", s=15)

anim = animation.FuncAnimation(
    fig,
    update,
    init_func=init,
    frames= range(1,len(xvar) * len(yvar)),
    interval=1)

# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
# anim.save('circle_flight.mp4', writer=writer)

plt.show(block=False)