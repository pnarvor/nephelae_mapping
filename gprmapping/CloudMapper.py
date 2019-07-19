import numpy as np

from sklearn.gaussian_process import kernels as gpk
# from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

class CloudMapper:

    """CloudMapper

    Main class producing cloud maps using Gaussian Process Regression.

    Handles :
        - Production of dense cloud maps (on request ?, auto generate ?)
        - Management of trainning data :
            - Filtering of trainning data (for efficiency)
            - Time partition of trainning data (for efficiency)
            - Producer partition of trainning data (for reasons).
            - Can subscribe to a nephelae_mapping.database.NephelaeDataServer
              to get stream of data (method add_sample ?).
            - Can directly query data from the dataServer.
    """

    def __init__(self, kernel, variableName, dataServer):

        self.variableName = variableName
        self.kernel       = kernel
        self.dataServer   = dataServer




