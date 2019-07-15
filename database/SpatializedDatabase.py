# TODO TODO TODO TODO TODO TODO TODO
# Search for a dedicated library to do this
# Go search arounf pysqlite
# TODO TODO TODO TODO TODO TODO TODO

import numpy as np
import bisect as bi
import pickle
import os
import threading


class SpbEntry:

    """SpbEntry

    Aim to unify the elements in the SpatializedDatabase.
    Contains a space-time location and at least one tag.

    """

    def __init__(self, data, position, tags=['misc']):

        self.data     = data
        self.position = position
        self.tags     = tags


class SpbSortableElement:

    """SpbSortableElement

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
        if isinstance(other, SpbSortableElement):
            return self.index < other.index
        else:
            return self.index < other

    def __le__(self, other):
        if isinstance(other, SpbSortableElement):
            return self.index <= other.index
        else:
            return self.index <= other

    def __eq__(self, other):
        if isinstance(other, SpbSortableElement):
            return self.index == other.index
        else:
            return self.index == other

    def __ne__(self, other):
        if isinstance(other, SpbSortableElement):
            return self.index != other.index
        else:
            return self.index != other

    def __ge__(self, other):
        if isinstance(other, SpbSortableElement):
            return self.index >= other.index
        else:
            return self.index >= other

    def __gt__(self, other):
        if isinstance(other, SpbSortableElement):
            return self.index > other.index
        else:
            return self.index > other


class SpatializedList:

    """SpatializedList

    Class to (supposedly) efficiently insert and retrieve data based on their
    space-time location.

    Heavily based on python3 bisect module.

    All data element are assumed to have the same interface as SpbEntry

    Base principle is to keep 4 list containing the same data but sorted along
    each dimension of space time (seems an awful waste of memory but only
    references to data are duplicated, not the data itself).
    When a query is made, smaller lists are made from subsets of the main
    list and the result is the common elements between the smaller lists.


    """

    def __init__(self):

        self.tSorted = []
        self.xSorted = []
        self.ySorted = []
        self.zSorted = []


    def insert(self, data):

        # data assumed to be of a SpbEntry compliant type
        bi.insort(self.tSorted, SpbSortableElement(data.position.t, data))
        bi.insort(self.xSorted, SpbSortableElement(data.position.x, data))
        bi.insort(self.ySorted, SpbSortableElement(data.position.y, data))
        bi.insort(self.zSorted, SpbSortableElement(data.position.z, data))


    def process_keys(self, keys):
        
        if keys is None:
            return (slice(None), slice(None), slice(None), slice(None))
        
        keys = list(keys)
        while len(keys) < 4:
            keys.append(slice(None,None,None))

        def process_key(key, sortedList):
            if key is None:
                return None
            elif key < 0:
                return sortedList[-1].index + key
            else:
                return key
        
        return (slice(process_key(keys[0].start, self.tSorted),
                      process_key(keys[0].stop,  self.tSorted)),
                slice(process_key(keys[1].start, self.xSorted),
                      process_key(keys[1].stop,  self.xSorted)),
                slice(process_key(keys[2].start, self.ySorted),
                      process_key(keys[2].stop,  self.ySorted)),
                slice(process_key(keys[3].start, self.zSorted),
                      process_key(keys[3].stop,  self.zSorted)))


    def find_entries(self, tags=[], keys=None):

        """SpatializedList.__getitem__
        keys : a tuple of slices(float,float,None)
               slices values are bounds of a 4D cube in which are the
               requested data
               There must exactly be 4 slices in the tuple
        """

        print("input keys  :", keys)
        keys = self.process_keys(keys)
        print("output keys :", keys)

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


class SpatializedDatabase:

    """SpatializedDatabase

    This is a test class for Nephelae raw Uav data server.
    Must handle space-time related requests like all data in a region of 
    space-time. (Hence the very well though name TODO: find a real one). 
    Made to match the subscriber pattern used in nephelae_paparazzi.PprzUav.

    """

    # class member functions #####################################

    def serialize(database):
        return pickle.dump(self)


    def unserialize(stream):
        return pickle.load(stream)

    
    def load(path):
        return pickle.load(open(path, "rb"))


    def save(database, path, force=False):
        if not force and os.path.exists(path):
            raise ValueError("Path \"" + path + "\" already exists. "
                             "Please delete the file, pick another path "
                             "or force overwritting with force=True")
        pickle.dump(database, open(path + '.part', "wb"))
        # this for not to erase the previously saved database in case of failure
        os.rename(path + '.part', path)


    # instance member functions #################################

    def __init__(self):
        self.taggedData = {'ALL': SpatializedList()}
        self.saveTime = None


    def insert(self, entry):
        self.taggedData['ALL'].insert(entry)
        for tag in entry.tags:
            if tag not in self.taggedData.keys():
                self.taggedData[tag] = SpatializedList()
            self.taggedData[tag].insert(entry)


    def find_entries(self, tags=[], keys=None):
        if not tags:
            return self.taggedData['ALL'].find_entries(keys=keys)
        else:
            return self.taggedData[tags[0]].find_entries(tags, keys)


    def __getstate__(self):
        return self.taggedData
  

    def __setstate__(self, taggedData):
        self.taggedData = taggedData


    def enable_periodic_save(self, path, timerTick=60.0, force=False):
        if not force and os.path.exists(path):
            raise ValueError("Path \"" + path + "\" already exists. "
                             "Please delete the file, pick another path "
                             "or force overwritting with force=True")
        self.saveTimerTick = timerTick
        self.savePath      = path
        self.saveTimer = threading.Timer(self.saveTimerTick,
                                         self.periodic_save_do)
        self.saveTimer.start()


    def disable_periodic_save(self):
        if self.saveTimer is not None:
            self.saveTimer.cancel()
        self.saveTimer = None

    
    def periodic_save_do(self):
        SpatializedDatabase.save(self, self.savePath, force=True)
        if self.saveTimer is not None: # check if disable was called
            self.saveTimer = threading.Timer(self.saveTimerTick,
                                             self.periodic_save_do)
            self.saveTimer.start()





