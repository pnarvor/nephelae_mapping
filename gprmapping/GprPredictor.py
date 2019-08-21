import numpy as np

from nephelae_base.types import Bounds
from sklearn.gaussian_process import GaussianProcessRegressor

from .MapInterface import MapInterface

class GprPredictor(MapInterface):

    """GprPredictor

    Class dedicated to prediction using Gaussian Process Regression.

    No kernel parameters is optimised, kernel is given as a fixed parameter.

    """

    def __init__(self, variableName, database, databaseTags, kernel):

        """
        variableName (str):
            name of the variable (no inside class purpose, only an identifier)

        database (nephelae_mapping.database):
            database from which fetching the measured data

        databaseTags (list of strings):
            tags for searching data in the database
       
        kernel (GprKernel): kernel to use for the prediction
                            (is compatiable with scikit-learn kernels)
        """
        super().__init__(variableName)

        self.database     = database
        self.databaseTags = databaseTags
        self.kernel       = kernel
        self.gprProc = GaussianProcessRegressor(self.kernel,
                                                alpha=0.0,
                                                optimizer=None,
                                                copy_X_train=False)


    def at_locations(self, locations, returnStddev=True):

        # ############## WRRRROOOOOOOOOOOOOOOOOOOOOOONNG #####################
        # # Must take all data otherwise prediction not possible because outside 
        # # locations
        # searchKeys = [slice(b.min,b.max) for b in Bounds.from_array(locations.T)]
        # samples = [entry.data for entry in \
        #            self.database.find_entries(self.databaseTags, tuple(searchKeys)]

        # Finding bounds of data we currently have then compute
        # proper bounds of data to request. This ensure we have some data
        # to make predictions on.
        # Here only time limts are considered
        # locations are assumed to be sorted in increasing time order
        # (same time location will probably be asked more often anyway)
        dataBounds = self.database.find_bounds(self.databaseTags)[0]
        kernelSpan = self.kernel.span()[0]
        locBounds = Bounds(locations[0,0], locations[-1,0])

        locBounds.min = max([locBounds.min, dataBounds.min])
        locBounds.max = min([locBounds.max, dataBounds.max])
        locBounds.min = locBounds.min - kernelSpan
        locBounds.max = locBounds.max + kernelSpan

        samples = [entry.data for entry in \
            self.database.find_entries(self.databaseTags,
                                       (slice(locBounds.min, locBounds.max),))]
        
        trainLocations =\
            np.array([[s.position.t,\
                       s.position.x,\
                       s.position.y,\
                       s.position.z]\
                       for s in samples])
        trainValues = np.array([s.data for s in samples])
        self.gprProc.fit(trainLocations, trainValues)
        return self.gprProc.predict(locations, return_std=returnStddev)


    def shape(self):
        return (None, None, None, None)


    def span(self):
        return (None, None, None, None)


    def bounds(self):
        return (None, None, None, None)


    def resolution(self):
        return self.kernel.resolution()



