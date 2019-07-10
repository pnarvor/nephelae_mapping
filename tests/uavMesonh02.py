#! /usr/bin/python3

import sys
sys.path.append('../')
import os
import signal
import time

from ivy.std_api import *
import logging

import nephelae_paparazzi.pprzinterface as ppint
from database import Database

mesonhFiles = '/home/pnarvor/work/nephelae/data/MesoNH-2019-02/REFHR.1.ARMCu.4D.nc'

class Logger:

    def __init__(self):
        pass

    def add_sample(self, sample):
        print(sample, end="\n\n")

    def add_gps(self, gps):
        print(gps, end="\n\n")

dtbase = Database()
def build_uav(uavId, navRef):
    uav = ppint.PprzMesoNHUav(uavId, navRef, mesonhFiles, ['RCT', 'WT'])
    uav.add_sensor_observer(dtbase)
    uav.add_gps_observer(dtbase)
    
    # Uncomment this for data display
    # uav.add_sensor_observer(Logger())
    # uav.add_gps_observer(Logger())
    return uav

interface = ppint.PprzSimulation(mesonhFiles,
                                 ['RCT', 'WT'],
                                 build_uav_callback=build_uav)
# ### wind feedback only
# interface = ppint.PprzSimulation(mesonhFiles, [], build_uav_callback=None)
interface.start()
# Hase to be called after interface.start()
dtbase.set_navigation_frame(interface.navFrame)

def stop():
    if interface.running:
        print("Shutting down... ", end='')
        sys.stdout.flush()
        interface.stop()
        print("Complete.")
signal.signal(signal.SIGINT, lambda sig,fr: stop())

