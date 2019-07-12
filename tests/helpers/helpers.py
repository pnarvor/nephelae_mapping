class Logger:

    def __init__(self):
        pass

    def add_sample(self, sample):
        print(sample, end="\n\n")

    def add_gps(self, gps):
        print(gps, end="\n\n")


class Fancy:
    """
    Helper class to call DoubleBufferedCache.load() with fancy indexing
        - usage : yourArray(Fancy()[fancy_indexes])
        - Introduce overhead, mosly for convenient testing.
    """
    def __getitem__(self, keys):
        if isinstance(keys, tuple):
            return keys
        else:
            return (keys,)

