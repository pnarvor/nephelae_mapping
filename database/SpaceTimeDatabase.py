# TODO TODO TODO TODO TODO TODO TODO
# Search for a dedicated library to do this
# Go search arounf pysqlite
# TODO TODO TODO TODO TODO TODO TODO

import numpy as np
import bisect as bi

from nephelae_base.types import Position
from nephelae_base.types import SensorSample
from nephelae_base.types import MultiObserverSubject


class StbEntry:

    """StbEntry

    Aim to unify the elements in the SpaceTimeDatabase.
    Contains a space-time location and at least one tag.

    """

    def __init__(self, data, position, tags=['misc']):

        self.data     = data
        self.position = position
        self.tags     = tags


class StbSortableElement:

    """StbSortableElement

    Intended to be used as a generic container in sorted lists.
    Contains a single index value to use for sorting and a data sample.
    All the numerical comparison operators are overloaded to compare  only
    the indexes of two instances.

    TODO : see if already exists

    """

    def __init__(self, index, data):
        self.index = index
        self.data  = data

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '{'+str(self.index)+' '+str(self.data)+'}'

    def __lt__(self, other):
        if isinstance(other, StbSortableElement):
            return self.index < other.index
        else:
            return self.index < other

    def __le__(self, other):
        if isinstance(other, StbSortableElement):
            return self.index <= other.index
        else:
            return self.index <= other

    def __eq__(self, other):
        if isinstance(other, StbSortableElement):
            return self.index == other.index
        else:
            return self.index == other

    def __ne__(self, other):
        if isinstance(other, StbSortableElement):
            return self.index != other.index
        else:
            return self.index != other

    def __ge__(self, other):
        if isinstance(other, StbSortableElement):
            return self.index >= other.index
        else:
            return self.index >= other

    def __gt__(self, other):
        if isinstance(other, StbSortableElement):
            return self.index > other.index
        else:
            return self.index > other


class SpaceTimeList:

    """SpaceTimeList

    Class to (supposedly) efficiently insert and retrieve data based on their
    space-time location.

    Heavily based on python3 bisect module.

    All data element are assumed to have the same interface as StbEntry

    Base principle is to keep 4 list containing the same data but sorted along
    each dimension of space time (seems an awful waste of memory but only
    duplicate references to data are duplicated, not the data it self).
    When a query is made, smaller lists are made from subsets of the main
    list and the result is the common elements between the smaller lists.


    """

    def __init__(self):

        self.tSorted = []
        self.xSorted = []
        self.ySorted = []
        self.zSorted = []


    def insert(self, data):

        # data assumed to be of a StbEntry compliant type
        bi.insort(self.tSorted, StbSortableElement(data.position.t, data))
        bi.insort(self.xSorted, StbSortableElement(data.position.x, data))
        bi.insort(self.ySorted, StbSortableElement(data.position.y, data))
        bi.insort(self.zSorted, StbSortableElement(data.position.z, data))


    def find_entries(self, tags=[], keys=None):

        """SpaceTimeList.__getitem__
        keys : a tuple of slices(float,float,None)
               slices values are bounds of a 4D cube in which are the
               requested data
               There must exactly be 4 slices in the tuple
        """

        if keys is None:
            keys = (slice(None,None,None),
                    slice(None,None,None),
                    slice(None,None,None),
                    slice(None,None,None))
        
        # Supposedly efficient way
        # Using a python dict to remove duplicates
        outputDict = {}
        def extract_entries(sortedList, key, outputDict):
            if key.start is None:
                key_start = None
            else:
                key_start = bi.bisect_left(sortedList, key.start)
            if key.stop is None:
                key_stop = None
            else:
                key_stop = bi.bisect_right(sortedList, key.stop)
            slc = slice(key_start, key_stop, None)
            for element in sortedList[slc]:
                if all([tag in element.data.tags for tag in tags]):
                    if id(element.data) not in outputDict.keys():
                        outputDict[id(element.data)] = []
                    # Will insert if tags is empty (all returns True on empty list)
                    outputDict[id(element.data)].append(element.data)

        extract_entries(self.tSorted, keys[0], outputDict)
        extract_entries(self.xSorted, keys[1], outputDict)
        extract_entries(self.ySorted, keys[2], outputDict)
        extract_entries(self.zSorted, keys[3], outputDict)

        return [l[0] for l in outputDict.values() if len(l) == 4]


class SpaceTimeDatabase:

    """SpaceTimeDatabase

    This is a test class for Nephelae raw Uav data server.
    Must handle space-time related requests like all data in a region of 
    space-time. (Hence the very well though name TODO: find a real one). 
    Made to match the subscriber pattern used in nephelae_paparazzi.PprzUav.

    """

    def __init__(self):
        self.taggedData = {}


    def insert(self, entry):
        for tag in entry.tags:
            if tag not in self.taggedData.keys():
                self.taggedData[tag] = SpaceTimeList()
            self.taggedData[tag].insert(entry)


    def find_entries(self, tags=[], keys=None):
        if not tags:
            return self.taggedData.values()[0].find_entries(keys=keys)
        else:
            return self.taggedData[tags[0]].find_entries(tags, keys)


class NephelaeDatabase(SpaceTimeDatabase):

    """NephelaeDatabase

    Subclass of SpaceTimeDatabe for specialization for Nephelae project

    /!\ Find better name

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
        self.insert(StbEntry(gps, gps - self.navFrame, tags))


    def add_sample(self, sample):
        # sample assumed to comply with nephelae_base.types.sensor_sample
        self.observerSet.add_sample(sample)
        if self.navFrame is None:
            return
        self.samples.append(sample)
        tags=[str(sample.producer),
              str(sample.variableName),
              'SAMPLE']
        self.insert(StbEntry(sample, sample.position, tags))


    def add_gps_observer(self, observer):
        self.observerSet.attach_observer(observer, 'add_gps')


    def add_sensor_observer(self, observer):
        self.observerSet.attach_observer(observer, 'add_sample')




