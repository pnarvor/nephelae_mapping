#! /usr/bin/python3

import sys
sys.path.append('../../')
import os
import signal
import time

from ivy.std_api import *
import logging

from nephelae_mapping.database import NephelaeDataServer

from helpers.helpers import *


# Example loading a previously saved nephelae database
# see tests/database_save01.py to see how it was generated
dtbase = NephelaeDataServer.load('output/database01.neph')

# Database is read using tags and time-space keys (t,x,y,z) 
# Only data containing all the tags are taken from the database
# For now, all data are tagged with the uav id which generated them and
#   - 'GPS' for gps messages
#   - 'RCT' for RCT values read in mesonh
#   - 'WT'  for upwind values read from mesonh
# if no tags are given all data matching the time-space keys are returned
# if no keys are given all data matching the tags are returned

# For now, 'GPS' messages match the Gps type in nephelae_paparazzi.pprzinterface.messages.Gps
#   (will be changed to a new type in nephelae_base.types/Gps, hopefuilly with same interface)
# For now, Sensor messages ('RCT', 'WT') match the SensorSample type in nephelae_base.types.SensorSample

# The Fancy() class is just a convenience class to use python fancy indexing
# The effective output of
#     Fancy()[:,10:,:10,10:20]
# is
#     (slice(None), slice(10,None), slice(None,10), slice(10,20))
# (same type as what is used to read in numpy arrays or python lists
#     

# Example getting the first 10s of GPS data for the uav '101'
output = [entry.data for entry in dtbase.find_entries(['GPS','101'], Fancy()[0:10.0])]
for item in output:
    print(item)

# Example getting the last 10s of GPS data for the uav '101'
output = [entry.data for entry in dtbase.find_entries(['GPS','101'], Fancy()[-10.0:])]
for item in output:
    print(item)



