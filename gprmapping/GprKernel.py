import numpy as np
import sklearn.gaussian_process.kernels as Kernel

class GprKernel:

    """GprKernel
    Wrapper class around scikit learn kernels

    This is to add some helper attributes.
    (Mainly to deduce the ideal sampling frequency of the kernel).
    
    """

    def __init__(self, kernel):
        self.kernel = kernel
   

    def values(self, locations):
        origin = np.zeros(len(locations[0]))
        res = []
        for point in locations:
            res.append(self.kernel(np.vstack([point, origin]))[0,1])
        return np.array(res)


    def ideal_resolution(self):
        pass

