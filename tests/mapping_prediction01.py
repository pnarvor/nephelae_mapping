#! /usr/bin/python3

import sys
sys.path.append('../../')
import numpy as np
import numpy.fft as npfft
import matplotlib.pyplot as plt
from   matplotlib import animation
import time

from netCDF4 import MFDataset
from nephelae_simulation.mesonh_interface import MesoNHVariable
from nephelae_mapping.gprmapping import NephKernel
from nephelae_mapping.gprmapping import GprPredictor
from nephelae_base.types import Position
from nephelae_base.types import Bounds


mesonhPath = '/home/pnarvor/work/nephelae/data/MesoNH-2019-02/REFHR.1.ARMCu.4D.nc'
rct = MesoNHVariable(MFDataset(mesonhPath), 'RCT')

kernel0 = NephKernel([20.0,50.0,50.0,50.0], noiseVariance = 0.01)

t = np.linspace(0,300.0,300)
a0 = 400.0
a1 = 0.0
f0 = 1 / 120.0
f1 = 1.5*f0

p0 = Position(240.0, 1700.0, 2000.0, 1100.0)
predNbPoints = 128
predBounds   = [Bounds(p0.x - 1.1*a0, p0.x + 1.1*a0),
                Bounds(p0.y - 1.1*a0, p0.y + 1.1*a0)]
pPredTmp = np.meshgrid(np.linspace(predBounds[0].min, predBounds[0].max, predNbPoints),
                       np.linspace(predBounds[1].min, predBounds[1].max, predNbPoints))
pPred = np.empty([predNbPoints**2, 4])
pPred[:,0] = p0.t
pPred[:,1] = pPredTmp[0].reshape(predNbPoints**2)
pPred[:,2] = pPredTmp[1].reshape(predNbPoints**2)
pPred[:,3] = p0.z

p  = np.array([[p0.t, p0.x, p0.y, p0.z]]*len(t))
p[:,1] = p[:,1] + a0*(a1 + np.cos(2*np.pi*f1*t))*np.cos(2*np.pi*f0*t)
p[:,2] = p[:,2] + a0*(a1 + np.cos(2*np.pi*f1*t))*np.sin(2*np.pi*f0*t)

rctValues = []
for pos in p:
    rctValues.append(rct[pos[0],pos[3],pos[2],pos[1]])
rctValues = np.array(rctValues)

predictor0 = GprPredictor(p, rctValues, kernel0)
prediction = predictor0(pPred)
# predValues = predictor0(pPred)[0].reshape([predNbPoints, predNbPoints])


fig, axes = plt.subplots(1,1)
b = rct.bounds
axes.imshow(rct[240.0,1100.0,:,:].data, origin='lower',
            extent=[b[3].min, b[3].max, b[2].min, b[2].max])
axes.plot(p0.x, p0.y, 'o')
axes.plot(p[:,1], p[:,2])

# fig, axes = plt.subplots(1,1)
# axes.plot(t[1:], np.linalg.norm(p[1:,1:3] - p[:-1, 1:3], axis=1))
# axes.grid()
# # axes.legend(loc="upper right")
# axes.set_xlabel("Time (s)")
# axes.set_ylabel("Uav velocity (m/s)")

fig, axes = plt.subplots(1,1)
axes.plot(t, rctValues)
axes.grid()
# axes.legend(loc="upper right")
axes.set_xlabel("Time (s)")
axes.set_ylabel("Uav velocity (m/s)")

predExtent = [predBounds[0].min, predBounds[0].max,
              predBounds[1].min, predBounds[1].max]
fig, axes = plt.subplots(1,2,sharex=True,sharey=True)
axes[0].imshow(prediction[0].reshape([predNbPoints, predNbPoints]),
               label='predicted rct value', origin='lower', extent=predExtent)
axes[0].grid()
# axes[0].legend(loc="upper right")
axes[0].set_xlabel("East (m)")
axes[0].set_ylabel("North (m)")
axes[1].imshow(prediction[1].reshape([predNbPoints, predNbPoints]),
               label='predicted rct cov', origin='lower', extent=predExtent)
axes[1].grid()
# axes[1].legend(loc="upper right")
axes[1].set_xlabel("East (m)")
axes[1].set_ylabel("North (m)")

fig, axes = plt.subplots(1,2,sharex=True,sharey=True)
axes[0].imshow(prediction[0].reshape([predNbPoints, predNbPoints]),
               label='predicted rct value', origin='lower', extent=predExtent)
axes[0].grid()
# axes[0].legend(loc="upper right")
axes[0].set_xlabel("East (m)")
axes[0].set_ylabel("North (m)")
axes[1].imshow(rct[p0.t,p0.z,predBounds[1].min:predBounds[1].max, predBounds[0].min:predBounds[0].max].data,
               label='ground truth', origin='lower', extent=predExtent)
axes[1].grid()
# axes[1].legend(loc="upper right")
axes[1].set_xlabel("East (m)")
axes[1].set_ylabel("North (m)")

plt.show(block=False)





