#! /usr/bin/python3

import sys
sys.path.append('../')
# import bisect
from bisect import *

from database import StbSortableElement

class DataStruct:

    def __init__(self, value='Hello !'):
        self.value = value

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.value)


l = [StbSortableElement(0.0, DataStruct(0)),
     StbSortableElement(3.0, DataStruct(3)),
     StbSortableElement(2.0, DataStruct(2)),
     StbSortableElement(1.0, DataStruct(1)),
     StbSortableElement(4.0, DataStruct(4))]
l.append(StbSortableElement(1.5, l[3].data))


print("Non sorted :")
print(l)

l.sort()
print("Sorted :")
print(l)

print("Numerical comparison elem 1 and 2:")
print(l[1] == l[2])

print("Data reference comparison elem 1 and 2:")
print(l[1].data is l[2].data)

insort(l, StbSortableElement(1.6, l[1].data))
print("bisect sorted insert")
print(l)

