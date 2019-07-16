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

print("loading database... ", end='', flush=True)
t0 = time.time()
# dtbase = NephelaeDataServer.load('output/database_perf01.neph')
dtbase = NephelaeDataServer.load('output/database_perf02.neph')
t1 = time.time()
print("Done. (ellapsed : ", t1 - t0,")", flush=True)


print("Reading database... ", flush=True)
t0 = time.time()
for i in range(10):
    output = [entry.data for entry in dtbase.find_entries(['GPS','101'],
                                        Fancy()[0:10.0,0:10.0,0:10.0,0:10.0])]
    # for item in output:
    #      print(item)
t1 = time.time()
print("Done. (ellapsed : ", t1 - t0,")", flush=True)



