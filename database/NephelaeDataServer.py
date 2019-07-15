import threading
import time

from nephelae_base.types import Position
from nephelae_base.types import SensorSample
from nephelae_base.types import MultiObserverSubject

from .SpatializedDatabase import SpatializedDatabase
from .SpatializedDatabase import SpbEntry

# from nephelae_base.types import ObserverSubject


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


class DatabasePlayer(NephelaeDataServer):

    """DatabasePlayer
    
    Class to replay messages stored in a NephelaeDataServer save.
    
    As a subclass of NephelaeDataServer it can be used as a dataserver for
    mapping and inteface testing.

    """

    def __init__(self, databasePath, timeFactor=1.0, granularity=0.005):
        super().__init__()
        
        self.origin       = SpatializedDatabase.load(databasePath)
        self.timeFactor   = timeFactor
        self.granularity  = granularity
        self.running      = False
        self.currentTime  = 0.0
        self.replayData   = []
        self.replayThread = None
        self.replayLock   = threading.Lock()

        self.set_navigation_frame(self.origin.navFrame)


    def play(self):
        if not self.running:
            self.restart()
        else:
            print("Replay already running.",
                  "Call 'restart' if you want to start it again")

    def stop(self):
        if self.running and self.replayThread is not None:
            print("Stopping replay... ", end='')
            self.running = False
            self.replayThread.join()
            print("Done.")


    def restart(self):
        if self.running:
            self.init_replay()
        else:
            self.replayThread = threading.Thread(target=self.run)
            self.replayThread.start()


    def init_replay(self):
        with self.replayLock:
            self.currentTime = 0.0
            sourceList = self.origin.taggedData['ALL'].tSorted
            self.replayData = [entry for entry in sourceList]

    
    def run(self):
        self.init_replay()
        lastTime = time.time()
        self.running = True
        while self.running and self.replayData:
            with self.replayLock:
                ellapsed = time.time() - lastTime
                self.currentTime = self.currentTime + self.timeFactor*ellapsed
                while self.replayData[0].index <= self.currentTime:
                    self.process_replayed_entry(self.replayData[0].data)
                    self.replayData.pop(0)
                    if not self.replayData:
                        break

            lastTime = lastTime + ellapsed
            time.sleep(self.granularity)
        self.running = False


    def process_replayed_entry(self, entry):
        if 'GPS' in entry.tags: 
            self.add_gps(entry.data)
        elif 'SAMPLE' in entry.tags:
            self.add_sample(entry.data)
        else:
            raise ValueError("No GPS or SAMPLE tag found in entry."+
                             "Are you using a valid database ?")



