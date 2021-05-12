import rasterio
import ground_truth
import pandas as pd
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from rasterio.io import MemoryFile

def raster_to_array(file, crop=False):
    img = rasterio.open(file)
    return get_training_array(img) if crop else img.read()


def get_training_array(tif):
    array = tif.read(window=from_bounds(*ground_truth.data.labelled_bounds + (tif.transform,)),
                     out_shape=(tif.count, ground_truth.data.height, ground_truth.data.width),
                     resampling=Resampling.bilinear)
    return normalise(array, tif.nodata)


def normalise(array, nodata):
    """Sets pixels with nodata value to zero then normalises each channel to between 0 and 1"""
    array[array == nodata] = 0
    return (array - array.min(axis=(1, 2))[:, None, None]) / (
        (array.max(axis=(1, 2)) - array.min(axis=(1, 2)))[:, None, None])

  
def resample_tif(tif, scale_factor, mode=Resampling.bilinear):
    data = tif.read(
        out_shape=(
            tif.count,
            int(tif.height * scale_factor),
            int(tif.width * scale_factor)
        ),
        resampling=mode
    )
    # scale image transform
    transform = tif.transform * tif.transform.scale(
        (tif.width / data.shape[-1]),
        (tif.height / data.shape[-2])
    )
    profile = tif.profile
    profile.update({'width': int(tif.width * scale_factor),
                    'height': int(tif.height * scale_factor),
                    'transform': transform})
    memory_file = MemoryFile().open(**profile)
    memory_file.write(data)
    return memory_file

import pandas as pd
def raster_to_df(file, value='class', single = True, crop = False):
    arr = ip.raster_to_array(file, crop)
    if single: df = pd.DataFrame(arr[0]).stack().rename_axis(['y', 'x']).reset_index(name=value).set_index(['y','x'])
    else: df = pd.DataFrame(arr.reshape(len(arr),-1)).stack().rename_axis(['y', 'x']).reset_index(name=value).set_index(['y','x'])
    return df, arr