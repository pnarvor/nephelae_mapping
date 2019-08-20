import os
import numpy as np
import numpy.fft as npfft
import pickle
import sklearn.gaussian_process.kernels as gpk

class NephKernel(gpk.Kernel):

    """NephKernel

    Base class for kernel used in the nephelae_project

    Kernel compatible with sklearn.gaussian_process.Kernel
    to be used in GaussianProcessRegressor.

    Is equivalent to a scikit-learn RBF kernel with white noise.
    
    /!\ Hyper parameters optimizatin HAS NOT BEEN TESTED
    When using with GaussianProcessRegressor, set optimizer=None

    Mostly to implement resolution and span methods.

    """

    def __init__(self, lengthScale, variance, noiseVariance):

        self.lengthScale   = lengthScale
        self.variance      = variance
        self.noiseVariance = noiseVariance


    def __call__(self, X, Y=None):

        if Y is None:
            Y = X

        distMat = cdist(X / self.lengthScale,
                        Y / self.lengthScale,
                        metric='sqeuclidian')
        if Y is X:
            return self.variance*np.exp(-0.5*distMat)\
                   + np.diag([self.noiseVariance]*X.shape[0])
        else:
            return self.variance*np.exp(-0.5*distMat)


    def diag(self, X):
        return np.array([self.variance + self.noiseVariance]*X.shape[0])


    def is_stationary(self):
        return True


    def resolution(self):
        # Value computed by estimating the cutting frequency of the RBF kernel
        # The kernel is the autocorrelation of the process and thus its fourier
        # transform is the power spectrum of the process.
        # Resolution is then computed as the inverse of twice the frequency
        # where the spectrum falls below -60db (shannon's theorem)
        return 0.84 * np.array(self.lengthScales)


    def span(self):
        # Distance from which a sample is deemed negligible
        return 3.0 * np.array(self.lengthScales)



class WindKernel(NephKernel):

    """WindKernel
    /!\ Only implemented for dimension (t,x,y,z). (only (t,x,y) won't work)
    """

    # Actually used (maybe)
    def __init__(self, lengthScale, variance, noiseVariance, windMap):
        """

        windMap : A MapInterface instance returning xy wind values.
        """
        super().__init__(lengthScale, variance, noiseVariance)
        self.windMap = windMap

    
    def __call__(self, X, Y=None):

        if Y is None:
            Y = X

        # print("X shape: ", X.shape)
        # print("Y shape: ", X.shape, end="\n\n")

        wind = self.windMap.at_locations(Y)

        # Far from most efficient but efficiency requires C++ implementation (or is it ?)
        t0,t1 = np.meshgrid(X[:,0], Y[:,0], indexing='ij', copy=False)
        dt = t1 - t0
        distMat = (dt / self.lengthScale[0])**2

        x0,x1 = np.meshgrid(X[:,1],    Y[:,1], indexing='ij', copy=False)
        x0,w1 = np.meshgrid(X[:,1], wind[:,0], indexing='ij', copy=False)
        dx = x1 - (x0 + w1 * dt)
        distMat = distMat + (dx / self.lengthScale[1])**2

        x0,x1 = np.meshgrid(X[:,2],    Y[:,2], indexing='ij', copy=False)
        x0,w1 = np.meshgrid(X[:,2], wind[:,1], indexing='ij', copy=False)
        dx = x1 - (x0 + w1 * dt)
        distMat = distMat + (dx / self.lengthScale[2])**2

        distMat = distMat + cdist(X[:,2] / self.lengthScale[3],
                                  Y[:,2] / self.lengthScale[3],
                                  metric='sqeuclidian')

        if Y is X:
            return self.variance*np.exp(-0.5*distMat)\
                   + np.diag([self.noiseVariance]*X.shape[0])
        else:
            return self.variance*np.exp(-0.5*distMat)




# class NephKernel(GprKernel):
# 
#     """NephKernel
# 
#     Specialization of GprKernel for Nephelae project use.
#     (for convenience, no heavy code here)
#     """
# 
#     def load(path):
#         return pickle.load(open(path, "rb"))
# 
# 
#     def save(kernel, path, force=False):
#         if not force and os.path.exists(path):
#             raise ValueError("Path \"" + path + "\" already exists. "
#                              "Please delete the file, pick another path "
#                              "or force overwritting with force=True")
#         pickle.dump(kernel, open(path, "wb"))
# 
# 
#     def __init__(self, lengthScales, variance, noiseVariance):
#         self.lengthScales  = lengthScales
#         self.noiseVariance = noiseVariance
#         self.variance      = variance
#         super().__init__(variance*gpk.RBF(lengthScales) + gpk.WhiteKernel(noiseVariance),
#                          lengthScales)
# 
# 
#     def __getstate__(self):
#         serializedItems = {}
#         serializedItems['lengthScales']  = self.lengthScales
#         serializedItems['noiseVariance'] = self.noiseVariance
#         return serializedItems
# 
# 
#     def __setstate__(self, data):
#         self.__init__(data['lengthScales'],
#                       data['noiseVariance'])




