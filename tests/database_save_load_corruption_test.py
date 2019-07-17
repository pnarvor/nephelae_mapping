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

# Example generating and saving a nephelae database #####################################
# A paparazzi simulation must be launched for this script to effectivly start
# paparazzi simulation (not necessarily launched before this script
# use ctrl-c or send a SIGINT to stop (stoping take sa few seconds)

database = NephelaeDataServer()
# /!\ An output folder named './output' must exists before hand
outputPath = 'output/database03.neph' 
database.enable_periodic_save(outputPath, timerTick=60.0, force=True)

# # uncomment this for feedback display (makes command unusable)
# logger = Logger()
# database.add_sensor_observer(logger)
# database.add_gps_observer(logger)

def build_uav(uavId, navRef):
    uav = ppint.PprzMesoNHUav(uavId, navRef, mesonhFiles, ['RCT', 'WT'])
    uav.add_sensor_observer(database)
    uav.add_gps_observer(database)
    
    return uav


interface = ppint.PprzSimulation(mesonhFiles,
                                 ['RCT', 'UT', 'VT', 'WT'],
                                 build_uav_callback=build_uav)
interface.start()
# Has to be called after interface.start()
database.set_navigation_frame(interface.navFrame)

def check_saved_database():
    def sorted_lists_equal(l0, l1):
        if len(l0) != len(l1):
            print("Lists have different lengths.")
        if not all([e0.index == e1.index and e0.data == e1.data for e0, e1 in zip(l0, l1)]):
            return False
        else:
            return True

    print("Checking database ... (may take some time)")
    reloaded = NephelaeDataServer.load(outputPath)
    print("Checking tags... ", end='')
    if database.taggedData.keys() != reloaded.taggedData.keys():
        raise ValueError("tag reload failed !")
    print("ok")
    for tag in database.taggedData.keys():
        print("Checking tag", tag, "... ")

        print("    Comparing tSorted lists... ", end='')
        l0 = database.taggedData[tag].tSorted
        l1 = reloaded.taggedData[tag].tSorted
        if not sorted_lists_equal(l0,l1):
            raise ValueError("tSorted lists not equal")
        print("ok")

        print("    Comparing xSorted lists... ", end='')
        l0 = database.taggedData[tag].xSorted
        l1 = reloaded.taggedData[tag].xSorted
        if not sorted_lists_equal(l0,l1):
            raise ValueError("xSorted lists not equal")
        print("ok")

        print("    Comparing ySorted lists... ", end='')
        l0 = database.taggedData[tag].ySorted
        l1 = reloaded.taggedData[tag].ySorted
        if not sorted_lists_equal(l0,l1):
            raise ValueError("ySorted lists not equal")
        print("ok")

        print("    Comparing zSorted lists... ", end='')
        l0 = database.taggedData[tag].zSorted
        l1 = reloaded.taggedData[tag].zSorted
        if not sorted_lists_equal(l0,l1):
            raise ValueError("zSorted lists not equal")
        print("ok")

    
    print("Done !")

def stop():
    if interface.running:
        print("Shutting down... ", end='')
        sys.stdout.flush()
        interface.stop()
        database.disable_periodic_save()
        print("Complete.")
signal.signal(signal.SIGINT, lambda sig,fr: stop())

