from nephelae_base.types import Position
from nephelae_base.types import SensorSample
from nephelae_base.types import MultiObserverSubject

from .SpatializedDatabase import SpatializedDatabase
from .SpatializedDatabase import SpbEntry


class NephelaeDataServer(SpatializedDatabase):

    """NephelaeDatabase

    SpatializedDatabase specialization for Nephelae project

    /!\ Find better name ?

    """

    def __init__(self):
        super().__init__() 

        self.navFrame = None
        self.observerSet = MultiObserverSubject(['add_gps', 'add_sample'])
       
        # For debug, to be removed
        self.gps      = []
        self.samples  = []


    def set_navigation_frame(self, navFrame):
        self.navFrame = navFrame


    def add_gps(self, gps):
        self.observerSet.add_gps(gps)
        if self.navFrame is None:
            return
        self.gps.append(gps)
        tags=[str(gps.uavId), 'GPS']
        self.insert(SpbEntry(gps, gps - self.navFrame, tags))


    def add_sample(self, sample):
        # sample assumed to comply with nephelae_base.types.sensor_sample
        self.observerSet.add_sample(sample)
        if self.navFrame is None:
            return
        self.samples.append(sample)
        tags=[str(sample.producer),
              str(sample.variableName),
              'SAMPLE']
        self.insert(SpbEntry(sample, sample.position, tags))


    def add_gps_observer(self, observer):
        self.observerSet.attach_observer(observer, 'add_gps')


    def add_sensor_observer(self, observer):
        self.observerSet.attach_observer(observer, 'add_sample')


    def __getstate__(self):
        return [self.navFrame, super().__getstate__()]
  

    def __setstate__(self, data):
        self.navFrame = data[0]
        super().__setstate__(data[1])

