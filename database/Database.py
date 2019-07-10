import numpy as np
from nephelae_base.types import SensorSample


class Database:

    """Database

    This is a test class for Nephelae raw Uav data server
    (Hence the very well though name TODO: find a real one). 
    Made to match the subscriber pattern used in nephelae_paparazzi.PprzUav.

    """

    def __init__(self):
        
        self.navFrame = None
        self.gps      = []
        self.samples  = []


    def set_navigation_frame(self, navFrame):
        self.navFrame = navFrame


    def add_gps(self, msg):
        if self.navFrame is None:
            return
        self.gps.append(msg)


    def add_sample(self, msg):
        if self.navFrame is None:
            return
        self.samples.append(msg)
        


