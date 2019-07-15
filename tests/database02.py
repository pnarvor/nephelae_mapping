#! /usr/bin/python3

import sys
sys.path.append('../../')
import os
import signal
import time

from ivy.std_api import *
import logging

import nephelae_paparazzi.pprzinterface as ppint
from nephelae_mapping.database import NephelaeDataServer

from helpers.helpers import *

mesonhFiles = '/home/pnarvor/work/nephelae/data/MesoNH-2019-02/REFHR.1.ARMCu.4D.nc'


dtbase = NephelaeDataServer()
dtbase.enable_periodic_save('output/database01.neph', timerTick=10.0)

# # uncomment this for feedback display (makes interpretor unusable)
# logger = Logger()
# dtbase.add_sensor_observer(logger)
# dtbase.add_gps_observer(logger)

def build_uav(uavId, navRef):
    uav = ppint.PprzMesoNHUav(uavId, navRef, mesonhFiles, ['RCT', 'WT'])
    uav.add_sensor_observer(dtbase)
    uav.add_gps_observer(dtbase)
    
    return uav


interface = ppint.PprzSimulation(mesonhFiles,
                                 ['RCT', 'WT'],
                                 build_uav_callback=build_uav)
interface.start()
# Has to be called after interface.start()
dtbase.set_navigation_frame(interface.navFrame)



def stop():
    if interface.running:
        print("Shutting down... ", end='')
        sys.stdout.flush()
        interface.stop()
        dtbase.disable_periodic_save()
        print("Complete.")
signal.signal(signal.SIGINT, lambda sig,fr: stop())

