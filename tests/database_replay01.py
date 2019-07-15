#! /usr/bin/python3

import sys
sys.path.append('../')
import os
import signal
import time

from ivy.std_api import *
import logging

from database import DatabasePlayer

from helpers.helpers import *


dtbase = DatabasePlayer('output/database01_10min.neph')

logger = Logger()
dtbase.add_gps_observer(logger)
dtbase.add_sensor_observer(logger)

dtbase.play()

signal.signal(signal.SIGINT, lambda sig, frame: dtbase.stop())

