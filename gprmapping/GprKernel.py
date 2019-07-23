import numpy as np
import numpy.fft as npfft
import sklearn.gaussian_process.kernels as Kernel

class GprKernel:

    """GprKernel
    Wrapper class around scikit learn kernels

    This is to add some helper attributes.
    (Mainly to deduce the ideal sampling frequency of the kernel).
    
    """

    def __init__(self, kernel, resolutions=None):
        self.kernel = kernel
        if resolutions is None:
            self.optimalResolutions = self.compute_optimal_resolutions()
        else:
            if len(resolutions) != len(self.kernel.get_params()['length_scale']):
                raise ValueError("resolutions paramater must the same dimension as the kernel")
            self.optimalResolutions = resolutions
   

    def values(self, locations):
        origin = np.zeros(len(locations[0]))
        res = []
        for point in locations:
            res.append(self.kernel(np.vstack([point, origin]))[0,1])
        return np.array(res)


    def compute_optimal_resolutions(self):
        # N = 4096 # find a criteria ? 
        lengthScaleFactor = 256
        # N = 512 # find a criteria ? 
        # N = 128 * lengthScaleFactor
        N = 32 * lengthScaleFactor
        resolutions = []

        lengthScales = self.kernel.get_params()['length_scale']
        if isinstance(lengthScales, float):
            lengthScales = [lengthScales]

        for dimIndex, lengthScale in enumerate(lengthScales):
            samplingRate = N / (2*lengthScaleFactor*lengthScale)

            # Generating a set of locations along the dimension to evaluate
            # (centered to simplify the fft evaluation
            locations = np.zeros([N, len(lengthScales)])
            locations[:,dimIndex] = np.linspace(-lengthScaleFactor*lengthScale,
                                                 lengthScaleFactor*lengthScale, N)

            # Evaluating the kernel values at these locations
            # (/!\ not the matrix k(X,X) /!\)
            kernelValues = self.values(locations)

            # Evaluating optimal resolution in Fourier space (Nyquist-Shannon theorem)
            ft = np.abs(npfft.fft(kernelValues))
            ftmin = min([a for a in ft if a > 0])
            ft[ft < ftmin] = ftmin
            ft = ft / np.max(ft) # max spectrum value will be 1.0, so 0.0 in db
            ft = 10*np.log10(ft) # spectrum in db

            # fc = (np.argmax(ft < -20) / N) * samplingRate == frequency where spectrum(kernel) < 20db
            # So optimal sampling frequency is f0 = 2.0*fc, hence optimal resolution is 1.0 / f0
            resolutions.append(1.0 / (2.0 * (np.argmax(ft < -20) / N) * samplingRate))
        return resolutions





