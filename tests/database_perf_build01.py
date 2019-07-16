#! /usr/bin/python3

import sys
sys.path.append('../../')
import os
import signal
import time
import random
import numpy as np
import matplotlib.pyplot as plt

from ivy.std_api import *
import logging

import nephelae_paparazzi.pprzinterface as ppint
import nephelae_paparazzi.pprzinterface.messages as pmsg
from nephelae_base.types import *
from nephelae_mapping.database import SpatializedDatabase
from nephelae_mapping.database import NephelaeDataServer

from helpers.helpers import *


dtbase = NephelaeDataServer()
# dtbase.enable_periodic_save('output/database_perf01.neph', 60.0, True)
dtbase.set_navigation_frame(pmsg.NavigationRef("100 NAVIGATION_REF 0.0 0.0 31 0.0"))
dtbase.navFrame.stamp = 0.0

uavIds   = ['100', '101', '102', '103', '104']
varNames = ['var0', 'var1', 'var2', 'var3', 'var4']
gpsSig  = 2000.0
dataSig = 10.0
N = 5*3600
t = []
try:
    t0 = time.time()
    for n in range(N):
    # for n in range(10):
        if int(100 * n / N) == 100.0*n / N:
            t1 = time.time()
            print("Filling database... ("+str(int(100*n/N))+"% : "+str(n)+"/"+str(N)+")")
            print(" - ", format(1000000.0*(t1 - t0) / (len(uavIds)*(len(varNames) + 1)*N) * 100.0, ".2f"),
                  "us per insert. (total : ", n*(len(uavIds)*(len(varNames) + 1)), "inserted)")
            t.append(1000000.0*(t1 - t0) / (len(uavIds)*(len(varNames) + 1)*N) * 100.0)
            t0 = t1
        for uavId in uavIds:
            gpsx = random.gauss(0.0, gpsSig)
            gpsy = random.gauss(0.0, gpsSig)
            gpsz = random.gauss(0.0, gpsSig)
            gps  = pmsg.Gps(uavId+" GPS 3 "+str(int(gpsx))+' '+str(int(gpsy))+' 0 '+str(int(gpsz))+
                       ' 0 0 0 0 31 0')
            gps.stamp = n
            dtbase.add_gps(gps)
    
            for var in varNames:
                sample = SensorSample(var, uavId, n, gps - dtbase.navFrame, 
                                      [random.gauss(0.0, dataSig)])
                dtbase.add_sample(sample)
    
finally:    
    fig, axes = plt.subplots(1,1)
    axes.plot(np.linspace(0, (len(uavIds)*(len(varNames) + 1)*N), len(t)) ,t, label="insert time")
    axes.set_xlabel("database size")
    axes.set_ylabel("insert time (us)")
    axes.grid()
    plt.show(block=False)
    # dtbase.disable_periodic_save()
    dtbase.save('output/database_perf02.neph', True)





