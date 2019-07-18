import numpy as np

from nephelae_base.types import Bounds

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels as gpk
# from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

class GprPredictor:

    """GprPredictor

    Class dedicated to prediction using Gaussian Process Regression.

    No kernel parameters is optimised, kernel is given as a fixed parameter.

    This class also keeps a kind of history of sensor data in the form of
    precomputed posterior distribution given the observation. Also keeps the
    observation, as they are needed for prediction.
    """

    def __init__(self, trainLocations, trainValues, kernel, noiseStd):

        """
        obsLocations : TxN np.array, T observations locations in a N-D space.
        obsValues    : Tx1 np.array, values of observed variable at obsLocations.
        kernel       : a sklearn.gaussian_process.kernels compliant kernel.

        """
        self.trainLocations = trainLocations
        self.trainValues    = trainValues
        self.bounds = [Bounds(m,M) for m,M in zip(trainLocations.min(axis=0),
                                                  trainLocations.max(axis=0))]
        self.gprProcessor = GaussianProcessRegressor(kernel=kernel,
                                                     alpha=noiseStd**2,
                                                     optimizer=None,
                                                     copy_X_train=False)
        self.gprProcessor.fit(trainLocations, trainValues.reshape(-1,1))


    def __call__(self, locations):
        return self.gprProcessor.predict(locations, return_std=True)
                                                     


