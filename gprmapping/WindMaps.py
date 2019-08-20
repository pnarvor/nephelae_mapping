import numpy as np

from nephelae_base.types import Bounds

from .MapInterface import MapInterface

class WindMapConstant(MapInterface):

    """WindConstant
    
    Constant wind predictor.
    
    Will output the same value at every point in space.
    """

    def __init__(self, variableName, wind=[0.0,0.0],
                 resolution=[50.0,50.0,50.0,50.0]):
        super().__init__(variableName)
        self.wind = wind
        self.resol = resolution


    def at_locations(self, locations):
        return np.array([self.wind]*locations.shape[0])


    def shape(self):
        return (None,None,None,None)


    def span(self):
        return (None,None,None,None)


    def bounds(self):
        return (None,None,None,None)


    def resolution(self):
        return self.resol
