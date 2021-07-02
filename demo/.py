import rasterio
class GroundTruth(object):
    def __init__(self, file):
        self.file = file
        tif = rasterio.open(file)
        self.labelled_bounds = tif.bounds
        self.height = tif.height
        self.width = tif.width
        self.resolution = tif.transform[0]
        self.array = tif.read()