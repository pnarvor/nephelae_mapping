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
from nephelae_base.types import Position
from nephelae_base.types import Bounds
from nephelae_base.types import Gps
from nephelae_base.types import SensorSample

from nephelae_mapping.gprmapping import GprPredictor
from nephelae_mapping.gprmapping import WindKernel
from nephelae_mapping.gprmapping import WindMapConstant
from nephelae_mapping.database   import NephelaeDataServer

from sklearn.gaussian_process import GaussianProcessRegressor

def parameters(rct):

    # Trajectory ####################################
    # a0 = 400.0
    a0 = 250.0
    f0 = - 1 / 120.0
    # f0 = 1 / 150.0
    
    a1 = 0.0
    # f1 = 1.5*f0
    f1 = 2.5*f0
    # f1 = -1.3*f0
    # f1 = -2.5*f0
    # f1 = -4.5*f0
    v0 = np.array([8.5, 0.9])
    
    tStart = 50.0
    tEnd   = 700.0
    t = np.linspace(tStart, tEnd, int(tEnd - tStart))
    # p0 = Position(240.0, 1700.0, 2000.0, 1100.0)
    # p0 = Position(50.0, 0.0, 2000.0, 1100.0)
    p0 = Position(50.0, 100.0, 1950.0, 1100.0)
    p  = np.array([[p0.t, p0.x, p0.y, p0.z]]*len(t))
    
    p[:,0] = t
    p[:,1] = p[:,1] + a0*(a1 + np.cos(2*np.pi*f1*(t-t[0])))*np.cos(2*np.pi*f0*(t-t[0]))
    p[:,2] = p[:,2] + a0*(a1 + np.cos(2*np.pi*f1*(t-t[0])))*np.sin(2*np.pi*f0*(t-t[0]))
    print("Max velocity relative to wind :",
        max(np.sqrt(np.sum((p[1:,1:3] - p[:-1,1:3])**2, axis=1)) / (p[1:,0] - p[:-1,0])))
    p[:,1:3] = p[:,1:3] + (t - tStart).reshape([len(t), 1]) @ v0.reshape([1,2]) 
    
    # prediction ####################################
    b = rct.bounds
    yBounds = [min(p[:,2]), max(p[:,2])]
    tmp = rct[p0.t,p0.z,yBounds[0]:yBounds[1],:]
    X0,Y0 = np.meshgrid(
        np.linspace(tmp.bounds[1][0], tmp.bounds[1][-1], tmp.shape[1]),
        np.linspace(tmp.bounds[0][0], tmp.bounds[0][-1], tmp.shape[0]),
        copy=False)
    xyLocations = np.array([[0]*X0.shape[0]*X0.shape[1], X0.ravel(), Y0.ravel()]).T
    b[2].min = yBounds[0]
    b[2].max = yBounds[1]

    return t,p0,p,b,xyLocations,v0,tmp.shape,tStart,tEnd

#########################################################################

mesonhPath = '/home/pnarvor/work/nephelae/data/MesoNH-2019-02/REFHR.1.ARMCu.4D.nc'
rct = MesoNHVariable(MFDataset(mesonhPath), 'RCT')
ut  = MesoNHVariable(MFDataset(mesonhPath), 'UT')
vt  = MesoNHVariable(MFDataset(mesonhPath), 'VT')

t,p0,p,b,xyLocations,v0,mapShape,tStart,tEnd = parameters(rct)

# Kernel
processVariance    = 1.0e-8
noiseStddev = 0.1 * np.sqrt(processVariance)
# kernel0 = WindKernel(lengthScales, processVariance, noiseStddev**2, v0)
# lengthScales = [70.0, 60.0, 60.0, 60.0]
lengthScales = [70.0, 80.0, 80.0, 60.0]
kernel0 = WindKernel(lengthScales, processVariance, noiseStddev**2, WindMapConstant('Wind',v0))

noise = noiseStddev*np.random.randn(p.shape[0])
dtfile = 'output/wind_data03.neph'
print("Getting mesonh values... ", end='')
# dtbase = NephelaeDataServer()
# sys.stdout.flush()
# for pos,n in zip(p,noise):
#     dtbase.add_gps(Gps("100", Position(pos[0],pos[1],pos[2],pos[3])))
#     dtbase.add_sample(SensorSample('RCT', '100', pos[0],
#         Position(pos[0],pos[1],pos[2],pos[3]),
#         [rct[pos[0],pos[3],pos[2],pos[1] + n]]))
#     dtbase.add_sample(SensorSample('Wind', '100', pos[0],
#         Position(pos[0],pos[1],pos[2],pos[3]),
#         [ut[pos[0],pos[3],pos[2],pos[1]], vt[pos[0],pos[3],pos[2],pos[1]]]))
# dtbase.save(dtfile, force=True)
dtbase = NephelaeDataServer.load(dtfile)
print("Done !")
sys.stdout.flush()

gprMap = GprPredictor('RCT', dtbase, ['RCT'], kernel0)


profiling = False
# profiling = True
if not profiling:
    fig, axes = plt.subplots(3,1,sharex=True,sharey=True)
simTime = p0.t
lastTime = time.time()
simSpeed = 50.0

def do_update(t):

    print("Sim time :", t)
    # prediction

    xyLocations[:,0] = t
    map0, std0 = gprMap.at_locations(xyLocations)
    map0[map0 < 0.0] = 0.0
    map0 = map0.reshape(mapShape)
    std0 = std0.reshape(mapShape)
   
    # display
    if not profiling:
        global axes
        axes[0].cla()
        axes[0].imshow(rct[t,p0.z,b[2].min:b[2].max,:].data, origin='lower',
                       extent=[b[3].min, b[3].max, b[2].min, b[2].max])
        axes[0].grid()
        axes[0].set_title("Ground truth")

        try:
            axes[0].plot(p[:int(t-tStart + 0.5),1], p[:int(t-tStart + 0.5),2], '.')
        finally:
            pass

        axes[1].cla()
        axes[1].imshow(map0, origin='lower',
                       extent=[b[3].min, b[3].max, b[2].min, b[2].max])
        axes[1].grid()
        axes[1].set_title("MAP")

        axes[2].cla()
        axes[2].imshow(std0**2, origin='lower',
                       extent=[b[3].min, b[3].max, b[2].min, b[2].max])
        axes[2].grid()
        axes[2].set_title("Variance AP")

def init():
    pass
def update(i):

    # global lastTime
    global simTime
    # currentTime = time.time()
    # simTime = simTime + simSpeed*(currentTime - lastTime)
    # lastTime = currentTime
    # simTime = simTime + 5.0
    simTime = simTime + 2.0

    do_update(simTime)

if not profiling:
    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        interval = 1)
    
    plt.show(block=False)
else:
    t0 = time.time()
    while simTime < 600:
        update(0)
    print("Ellapsed :", time.time() - t0, "s")

