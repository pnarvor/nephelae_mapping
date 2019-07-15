#! /usr/bin/python3

import os
import time
import numpy as np
import math as m
import matplotlib.pyplot as plt
from matplotlib import animation

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
tStart = 150 # initial timestamp

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

xi = 423
yi = 4300
s_w = 8 # wind speed
theta = 0 # angle(speed vector, x axis)
radius = 50
v_circle = 1200 # speed in circle
end_reach = False

def line_trajectory(xi, yi, t, theta=0, s=20):
    x_new = xi + s * np.cos(np.radians(theta)) * t
    y_new = yi + s * np.sin(np.radians(theta)) * t
    return x_new, y_new

def circ_trajectory(xc, yc, t, r, v=1000):
    w = v/r
    x_new = xc + r * np.cos(np.radians(w * t))
    y_new = yc + r * np.sin(np.radians(w * t))
    return x_new, y_new

def init():
    # axes[0][0].scatter(xi, yi, c="white", marker="x", s=20)
    plt.plot([xySlice.start, xySlice.stop], [y0, y0], color=[0.0, 0.0, 0.0, 1.0])
    #plt.scatter(xi, yi, c="white", marker="x", s=15)
    # axes[0][0].plot([xySlice.start, xySlice.stop], [y0, y0], color=[0.0, 0.0, 0.0, 1.0])
    # axes[1][0].plot([zSlice.start, zSlice.stop], [z0, z0], color=[0.0, 0.0, 0.0, 1.0])
    # axes[0][1].plot([xySlice.start, xySlice.stop], [y0, y0], color=[0.0, 0.0, 0.0, 1.0])
    # axes[1][1].plot([zSlice.start, zSlice.stop], [z0, z0], color=[0.0, 0.0, 0.0, 1.0])


def update(i):

    t = tStart+ i+ 1
    print(t)
    varDisp0.set_data(data0[t, z0, xySlice, xySlice].data)
    # varDisp1.set_data(data0[t, zSlice, y0, xySlice].data)
    # varDisp2.set_data(data1[t, z0, xySlice, xySlice].data)
    # varDisp3.set_data(data1[t, zSlice, y0, xySlice].data)

    global end_reach

    if (end_reach == False):
        xn, yn = circ_trajectory(xi, yi, t - tStart, radius, v_circle)
        xn, yn = line_trajectory(xn, yn, t - tStart, theta, s_w) # adding wind speed
        print(xn, yn)
        if (xn >= xyExtent[1] - 200 or yn >= xyExtent[3] - 200):
            end_reach = True
        else:
           plt.scatter(xn, yn, c="white", marker="x", s=15)

anim = animation.FuncAnimation(
    fig,
    update,
    init_func=init,
    frames=len(xvar) * len(yvar),
    interval=1)

plt.show(block=False)