# TODO TODO TODO TODO TODO TODO TODO
# Search for a dedicated library to do this
# TODO TODO TODO TODO TODO TODO TODO

import numpy as np
import bisect as bi

from nephelae_base.types import Position
from nephelae_base.types import SensorSample


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

    Container to insert and retrieve data based on their space-time location.

    All data element are assumed to have the same interface as StbEntry

    /!\ Changed to basic python implementation. To be continued

    """

    def __init__(self):

        self.data = []


    def insert(self, data):

        self.data.append(data)


    def __getitem__(self, keys):

        """SpaceTimeList.__getitem__
        keys : a tuple of slices(float,float,None)
               slices values are bounds of a 4D cube in which are the
               requested data
               There must exactly be 4 slices in the tuple
        """

        def isInSlice(value, key):
            if key.start is not None:
                if value < key.start:
                    return False
            if key.stop is not None:
                if value > key.stop:
                    return False
            return True
            return value >= key.start and value <= key.stop

        res = [item for item in self.data if isInSlice(item.position.t, keys[0])]
        res = [item for item in res if isInSlice(item.position.x, keys[1])]
        res = [item for item in res if isInSlice(item.position.y, keys[2])]
        res = [item.data for item in res if isInSlice(item.position.z, keys[3])]

        return res


class SpaceTimeDatabase:

    """SpaceTimeDatabase

    This is a test class for Nephelae raw Uav data server.
    Must handle space-time related requests like all data in a region of 
    space-time. (Hence the very well though name TODO: find a real one). 
    Made to match the subscriber pattern used in nephelae_paparazzi.PprzUav.

    """

    def __init__(self):
        
        self.navFrame = None
        self.gps      = []
        self.samples  = []
        self.


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
        


