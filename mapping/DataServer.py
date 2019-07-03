from nephelae_base.types import SensorSample

class DataServer:

    """DataServer

    Centralized data handler for mapping. All data producers directly
    notifies the DataServer when they get data.

    Attributes:

    data (dict): Hold all raw data samples.
                 Keys are variables names (pressure, temperature...).
                 A new key is automatically created when data with a new name
                 is received.

    Methods:

    notify(sample) : Add a sample to self.data. The sample must be compatible
                     with the SensorSample type. If the data type is not
                     in self.data.keys(), a new key is created.
    """

    def __init__(self):

        self.data = {}

    def add_sample(self, sample):
        
        if sample.variableName not in self.data.keys():
            self.data[sample.variableName] = []
        self.data[sample.variableName].append(sample)



