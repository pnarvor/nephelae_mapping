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


    def at_locations(self, locations):



        ############## WRRRROOOOOOOOOOOOOOOOOOOOOOONNG #####################
        # Must take all data otherwise prediction not possible because outside 
        # locations
        searchKeys = [slice(b.min,b.max) for b in Bounds.from_array(locations.T)]
        samples = [entry.data for entry in \
                   self.database.find_entries(self.databaseTags, tuple(searchKeys)]
        trainLocations =\
            np.array([[s.position.x, s.position.x, s.position.y, s.position.z] for s in samples])
        trainValues = np.array([s.data.data for s in samples])
        self.gprProc.fit(trainLocations, trainValues)




