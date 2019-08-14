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
    def __getitem__(self, keys):
        """
        return a slice of space filled with variable values.

        input:
            keys like reading a numpy.array (tuple of slice)

        output:
            numpy.array with values (squeezed in collapsed dimensions)
        """
        pass


    @abc.abstractmethod
    def shape():
        """
        List of number of data points in each dimensions.
        Can be empty if no dimensions, and element can be None
        if infinite dimension span
        """
        pass


    @abc.abstractmethod
    def span():
        """
        Returns a list of span of each dimension.
        Can be empty if no dimensions, and element can be None
        if infinite dimension span
        """
        pass


    @abc.abstractmethod
    def bounds():
        """
        Returns a list of bounds of each dimension.
        Can be empty if no dimensions, and element can be None
        if infinite dimension span
        """
        pass


    @abc.abstractmethod
    def resolution():
        """
        Return a list of resolution in each dimension
        Can be empty if no dimensions.
        Is ALWAYS defined for each dimension.
        """
        pass



