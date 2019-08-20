# abc = abstract class
import abc

class MapInterface(abc.ABC):

    """MapInterface

    This is an interface designed to be subclassed with GprPredictor class
    and MesoNHVariable. Its goal is to give a unified access for static
    MesoNH data in simulation or real time data estimation with flying uavs.

    """

    def __init__(self, variableName):

        self.variableName = variableName


    @abc.abstractmethod
    def at_locations(self, locations):
        """
        return variable value at locations

        input:
            locations: N*D np.array (N location, D dimensions)

        output:
            NxM np.array : variable value at locations
                           (variable is M dimensionnal)
        """
        pass


    @abc.abstractmethod
    def shape(self):
        """
        List of number of data points in each dimensions.
        Can be empty if no dimensions, and element can be None
        if infinite dimension span
        """
        pass


    @abc.abstractmethod
    def span(self):
        """
        Returns a list of span of each dimension.
        Can be empty if no dimensions, and element can be None
        if infinite dimension span
        """
        pass


    @abc.abstractmethod
    def bounds(self):
        """
        Returns a list of bounds of each dimension.
        Can be empty if no dimensions, and element can be None
        if infinite dimension span
        """
        pass


    @abc.abstractmethod
    def resolution(self):
        """
        Return a list of resolution in each dimension
        Can be empty if no dimensions.
        Is ALWAYS defined for each dimension.
        """
        pass


    def __getitem__(self, keys):
        """
        return a slice of space filled with variable values.

        input:
            keys like reading a numpy.array (tuple of slice)

        output:
            numpy.array with values (squeezed in collapsed dimensions)
        """

        params = []
        for key, res in zip(keys, self.resolution()):
            if isinstance(key, slice):
                size = int((key.stop - key.start)*res) + 1
                params.append(np.linspace(key.start, ker.start+(size-1)*res, size))
            else:
                params.append(key)
        T,X,Y,Z = np.meshgrid(params[0], params[1], params[2], params[3],
                              indexing='ij', copy=False)
        locations = np.array([T.ravel(), X.ravel(), Y.ravel(), Z.ravel()])

        # map0, std0 = self.at_locations(locations[np.argsort(locations[:,0])]) 
        # return map0.reshape(T.shape).squeeze(), std0.reshape(T,shape).squeeze()
        return self.at_locations(locations[np.argsort(locations[:,0])]).reshape(T.shape).squeeze()


