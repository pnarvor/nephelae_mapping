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

entries0 = database.find_entries(['101', 'WT'], Fancy()[-10::-1], sortCriteria=lambda x: x.data.position.t)
print("min:", entries0[0].data.position.t)
print("max:", entries0[-1].data.position.t)

entries1 = database.find_entries(['101', 'WT'], Fancy()[-10:], sortCriteria=lambda x: x.data.position.t)
print("min:", entries1[0].data.position.t)
print("max:", entries1[-1].data.position.t)





