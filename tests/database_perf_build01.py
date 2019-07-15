#! /usr/bin/python3

import sys
sys.path.append('../../')
import os
import signal
import time
import random

from ivy.std_api import *
import logging

import nephelae_paparazzi.pprzinterface as ppint
from nephelae_paparazzi.pprzinterface.messages import *
from nephelae_base.types import *
from nephelae_mapping.database import SpatializedDatabase
from nephelae_mapping.database import NephelaeDataServer

from helpers.helpers import *


dtbase = NephelaeDataServer()
# dtbase.enable_periodic_save('output/database_perf01.neph', 60.0, True)
dtbase.set_navigation_frame(NavigationRef("100 NAVIGATION_REF 0.0 0.0 31 0.0"))
dtbase.navFrame.stamp = 0.0

uavIds   = ['100', '101', '102', '103', '104']
varNames = ['var0', 'var1', 'var2', 'var3', 'var4']
gpsSig  = 2000.0
dataSig = 10.0
N = 5*3600
try:
    t0 = time.time()
    for n in range(N):
    # for n in range(10):
        if int(100 * n / N) == 100.0*n / N:
            t1 = time.time()
            print("Filling database... ("+str(int(100*n/N))+"% : "+str(n)+"/"+str(N)+")")
            print(" - ", format(1000.0*(t1 - t0) / (len(uavIds)*len(varNames)), ".2f"),
                  "ms per insert. (total : ", n, "inserted)")
            t0 = t1
        for uavId in uavIds:
            gpsx = random.gauss(0.0, gpsSig)
            gpsy = random.gauss(0.0, gpsSig)
            gpsz = random.gauss(0.0, gpsSig)
            gps  = Gps(uavId+" GPS 3 "+str(int(gpsx))+' '+str(int(gpsy))+' 0 '+str(int(gpsz))+
                       ' 0 0 0 0 31 0')
            gps.stamp = n
            dtbase.add_gps(gps)
    
            for var in varNames:
                sample = SensorSample(var, uavId, n, gps - dtbase.navFrame, 
                                      [random.gauss(0.0, dataSig)])
                dtbase.add_sample(sample)
    
finally:    
    # dtbase.disable_periodic_save()
    dtbase.save('output/database_perf02.neph', True)





