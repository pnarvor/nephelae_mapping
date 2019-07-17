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


database = NephelaeDataServer.load('output/database02.neph')

output = database.find_entries(['101', 'SAMPLE'], Fancy()[:60])
print("Got", len(output), "entries :") 
for entry in output:
    print(entry.data.position.t)


