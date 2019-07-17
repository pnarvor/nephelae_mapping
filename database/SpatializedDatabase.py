# TODO TODO TODO TODO TODO TODO TODO
# Search for a dedicated library to do this
# Go search arounf pysqlite
# TODO TODO TODO TODO TODO TODO TODO

import numpy as np
import bisect as bi
import pickle
import os
import threading

from nephelae_base.types import Bounds

class SpbEntry:

    """SpbEntry

    Aim to unify the elements in the SpatializedDatabase.
    Contains a space-time location and at least one tag.

    """

    def __init__(self, data, position, tags=['misc']):

        self.data     = data
        self.position = position
        self.tags     = tags


    def __eq__(self, other):
        if self.position != other.position:
            return False
        elif self.tags != other.tags:
            return False
        elif str(self.data) != str(other.data):
            return False
        return True


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


    def __len__(self):
        return len(self.tSorted)


    def insert(self, data):

        # data assumed to be of a SpbEntry compliant type
        bi.insort(self.tSorted, SpbSortableElement(data.position.t, data))
        bi.insort(self.xSorted, SpbSortableElement(data.position.x, data))
        bi.insort(self.ySorted, SpbSortableElement(data.position.y, data))
        bi.insort(self.zSorted, SpbSortableElement(data.position.z, data))


    def process_keys(self, keys):
        
        if keys is None:
            return (slice(None), slice(None), slice(None), slice(None))
        if len(keys) == 4: 
            return keys
        keys = list(keys)
        while len(keys) < 4:
            keys.append(slice(None))
        return tuple(keys)


    def build_entry_dict(self, tags=[], keys=None):

        """
        keys : a tuple of slices(float,float,None)
               slices values are bounds of a 4D cube in which are the
               requested data
               There must exactly be 4 slices in the tuple
        """
        
        keys = self.process_keys(keys)

        # Using a python dict to be able to remove duplicates
        # Supposedly efficient way
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
                    # Will insert if tags is empty (all([]) returns True)
                    outputDict[id(element.data)].append(element.data)

        extract_entries(self.tSorted, keys[0], outputDict)
        extract_entries(self.xSorted, keys[1], outputDict)
        extract_entries(self.ySorted, keys[2], outputDict)
        extract_entries(self.zSorted, keys[3], outputDict)

        return outputDict


    def find_entries(self, tags=[], keys=None, sortCriteria=None):

        """
        keys : a tuple of slices(float,float,None)
               slices values are bounds of a 4D cube in which are the
               requested data
               There must exactly be 4 slices in the tuple
        """
        
        outputDict = self.build_entry_dict(tags, keys)
        res = [l[0] for l in outputDict.values() if len(l) == 4]
        if sortCriteria is not None:
            res.sort(key=sortCriteria)
        return res


    def find_bounds(self, tags=[], keys=None):

        """
        keys : a tuple of slices(float,float,None)
               slices values are bounds of a 4D cube in which are the
               requested data
               There must exactly be 4 slices in the tuple
        """
        
        outputDict = self.build_entry_dict(tags, keys)
        bounds = [Bounds(), Bounds(), Bounds(), Bounds()]
        for l in outputDict.values():
            if len(l) != 4:
                continue
            bounds[0].update(l[0].data.position.t)
            bounds[1].update(l[0].data.position.x)
            bounds[2].update(l[0].data.position.y)
            bounds[3].update(l[0].data.position.z)
        return bounds


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
        self.saveTime = None
        self.init_data()
   

    def init_data(self):
        self.taggedData = {'ALL': SpatializedList()}
        self.orderedTags       = ['ALL']
        self.lastTagOrdering   = -1
        self.tagOrderingPeriod = 1000


    def insert(self, entry):
        self.taggedData['ALL'].insert(entry)
        for tag in entry.tags:
            if tag not in self.taggedData.keys():
                self.taggedData[tag] = SpatializedList()
            self.taggedData[tag].insert(entry)
        self.check_tag_ordering()


    def best_search_list(self, tags=[]):
        if not tags:
            return self.taggedData['ALL']
        else:
            for tag in self.orderedTags:
                if tag in tags:
                    return self.taggedData[tag]
            return self.taggedData['ALL']
        

    def find_entries(self, tags=[], keys=None, sortCriteria=None):
        # Making sure we have a list of tags, event with one element
        if isinstance(tags, str):
            tags = [tags]
        return self.best_search_list(tags).find_entries(tags, keys, sortCriteria)


    def find_bounds(self, tags=[], keys=None):
        # Making sure we have a list of tags, event with one element
        if isinstance(tags, str):
            tags = [tags]
        return self.best_search_list(tags).find_bounds(tags, keys)


    def __getstate__(self):
        return {'taggedData':self.taggedData}
  

    def __setstate__(self, deserializedData):
        self.init_data()
        self.taggedData = deserializedData['taggedData']
        self.check_tag_ordering()


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
        self.periodic_save_do()

    
    def periodic_save_do(self):
        SpatializedDatabase.save(self, self.savePath, force=True)
        if self.saveTimer is not None: # check if disable was called
            self.saveTimer = threading.Timer(self.saveTimerTick,
                                             self.periodic_save_do)
            self.saveTimer.start()

    
    def check_tag_ordering(self):
        if not 0 <= self.lastTagOrdering < self.tagOrderingPeriod:
            insertsSinceLinceLastTagOrdering = 0
            tags = []
            for tag in self.taggedData.keys():
                tags.append(SpbSortableElement(len(self.taggedData[tag]), tag))
            tags.sort()
            self.orderedTags = [tag.data for tag in tags]
        self.lastTagOrdering = self.lastTagOrdering + 1

