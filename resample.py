import rasterio

class hrCrop:
    def __init__(self):
        file_path = 'croptemp.tif'
        tif = rasterio.open(file_path)
        self.labelled_bounds = tif.bounds
        self.height = tif.height
        self.width = tif.width
        self.resolution = tif.transform[0]
        self.array = tif.read()

data = hrCrop()
