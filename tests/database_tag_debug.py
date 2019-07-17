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

import nephelae_paparazzi.pprzinterface as ppint
from nephelae_mapping.database import NephelaeDataServer

from helpers.helpers import *


database = NephelaeDataServer.load('output/database02.neph')

output = database.find_entries(['101', 'WT'], Fancy()[-3600:], sortCriteria=lambda x: x.data.timeStamp)
t  = np.array([entry.data.timeStamp for entry in output])
wt = np.array([entry.data.data[0] for entry in output])

fig, axes = plt.subplots(2,1)
axes[0].plot(t, label="time stamp")
axes[0].set_xlabel("Index")
axes[0].set_ylabel("Time (s)")
axes[0].legend(loc="lower right")
axes[0].grid()
axes[1].plot(t, wt, label="vertical wind")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Wind speed (m/s)")
axes[1].legend(loc="lower right")
axes[1].grid()

plt.show(block=False)


