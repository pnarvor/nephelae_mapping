#! /usr/bin/python3

import sys
sys.path.append('../../')
import os
import signal
import time
import numpy as np
import matplotlib.pyplot as plt

from ivy.std_api import *
import logging

from nephelae_base.types import Bounds
import nephelae_paparazzi.pprzinterface as ppint
from nephelae_mapping.database import NephelaeDataServer

from helpers.helpers import *


database = NephelaeDataServer.load('output/database02.neph')

bounds0 = database.find_bounds(['101', 'WT'], Fancy()[-3600:])
print("Queried bounds:", bounds0)
entries = database.find_entries(['101', 'WT'], Fancy()[-3600:])
bounds1 = [Bounds(), Bounds(), Bounds(), Bounds()]

for entry in entries:
    bounds1[0].update(entry.data.position.t)
    bounds1[1].update(entry.data.position.x)
    bounds1[2].update(entry.data.position.y)
    bounds1[3].update(entry.data.position.z)
print("Received bounds:", bounds1)





