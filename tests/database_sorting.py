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


database = NephelaeDataServer.load('output/database02.neph')
# output = database.find_entries(['GPS','101'], Fancy()[-60:])
output0 = database.find_entries(['GPS','101'], Fancy()[-60:])
output1 = database.find_entries(['GPS','101'], Fancy()[-60:],
                                sortCriteria=lambda x: x.data.stamp)
for entry0, entry1 in zip(output0, output1):
    print(format(entry0.data.stamp - database.navFrame.stamp, ".2f"),
          format(entry1.data.stamp - database.navFrame.stamp, ".2f"))


