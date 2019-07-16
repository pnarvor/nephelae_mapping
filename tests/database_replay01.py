#! /usr/bin/python3

import sys
sys.path.append('../../')
import os
import signal
import time

from ivy.std_api import *
import logging

from nephelae_mapping.database import DatabasePlayer

from helpers.helpers import *


dtbase = DatabasePlayer('output/database01.neph')

logger = Logger()
dtbase.add_gps_observer(logger)
dtbase.add_sensor_observer(logger)

dtbase.play(looped=True)

signal.signal(signal.SIGINT, lambda sig, frame: dtbase.stop())

