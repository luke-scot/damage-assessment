# Imports
import os
import importlib
import numpy as np
import pandas as pd
import rasterio as ro
import geopandas as gpd
import rioxarray as rxr
import shapely.geometry as sg
from glob import glob
from rasterio.merge import merge

#-------------------------------------------------------------#
"""Polygon Manipulation functions"""
def get_polygon(poly, conv = False):
    try: return sg.Polygon([[p['lat'], p['lng']] for p in poly.locations[0]]) if conv else sg.Polygon([[p['lng'], p['lat']] for p in poly.locations[0]])
    except: return sg.Polygon([[p[0],p[1]] for p in poly.locations]) if conv else sg.Polygon([[p[1],p[0]] for p in poly.locations])

def get_extent(poly, crsPoly='epsg:4326', crs=False, conv=False):
    newPoly = get_polygon(poly, conv=conv)
    if crs:
        bds = gpd.GeoSeries([newPoly], crs=crsPoly).to_crs(crs).bounds.values[0]
        return newPoly, gpd.GeoSeries([sg.Polygon.from_bounds(bds[0],bds[1],bds[2],bds[3])], crs={'init':crs})
    else: 
        return newPoly, gpd.GeoSeries([newPoly], crs={'init':crsPoly})

#-----------------------------------------------#
"""Shapefile import functions"""
# Function taking shapefile and outputting 
def shape_to_gdf(file, cn='class', splitString = False, crs=False):
    labels = gpd.read_file(file)
    if crs: labels=labels.to_crs({'init': crs})
    if cn not in labels.columns: raise ValueError(cn+" column not found in labels dataframe.")
    if splitString: labels[cn] = labels[cn].str.split(' ').str[0]
    return labels

  #----------------------------------------------#
"""Raster import functions"""
# Convert image raster to array
def raster_to_array(file, target=False, crop=False):
    img = ro.open(file)
    return get_training_array(img, target) if (crop is True and target is not False) else img.read()

# tif file to array with resampling crop file
def tif_to_array(tifFile, cropFile):
    cropFile = importlib.import_module(cropFile)
    importlib.reload(cropFile)
    tif, b = ro.open(tifFile), cropFile.data.labelled_bounds
    array = tif.read(window=ro.windows.from_bounds(b[0],b[1],b[2],b[3],transform=tif.transform),
                     out_shape=(tif.count, cropFile.data.height, cropFile.data.width),
                     resampling=ro.warp.Resampling.bilinear)
    tif.close()
    return array.copy()

# Array to dataframe for RGB image
def arr_to_df(arr):
    df = pd.DataFrame(arr.reshape(-1,len(arr)),columns=['R','G','B'])
    df['x'], df['y'] = np.tile(np.arange(arr.shape[2]),arr.shape[1]), np.repeat(range(arr.shape[1]),arr.shape[2])
    return highres.set_index(['y','x'])

# Function taking a raster input and outputting pandas dataframe
def raster_to_df(file, cn='class', target=False, crop = False):
    arr = raster_to_array(file, target=target, crop=crop)
    if arr.shape[0]>1:
        df = pd.DataFrame(arr.reshape(-1,len(arr)))
        df['x'], df['y'] = np.tile(np.arange(arr.shape[2]),arr.shape[1]), np.repeat(range(arr.shape[1]),arr.shape[2])
        df = df.set_index(['y','x'])
    else: df = pd.DataFrame(arr[0]).stack().rename_axis(['y', 'x']).reset_index(name=cn).set_index(['y','x'])
    return df, arr

# Similar to raster_to_df but with the option of specifying polygon for area of interest
def img_to_df(file, poly=False, crs=False, label='img', columns=False, crsPoly='epsg:4326', verbose=True):
    # Import raster
    img = rxr.open_rasterio(file, masked=True).squeeze()
    
    # Crop image if polygon supplied
    if poly:
        _, extent = get_extent(poly, crsPoly=crsPoly, crs=crs)
        img = img.rio.clip(extent.geometry.apply(sg.mapping))   
    named = img.rename('img')
    
    # Convert to dataframe
    xm, ym = np.meshgrid(np.array(named.coords['x']), np.array(named.coords['y']))
    mi = pd.MultiIndex.from_arrays([ym.flatten(),xm.flatten()],names=('y','x'))
    size = min(named.shape) if len(named.shape) > 2 else 1 
    df = pd.DataFrame(named.data.reshape(size,-1).transpose(), index=mi)
    if verbose: print(file+" read completed.")
    
    return df, named

#---------------------------------------------------#
"""Resampling Functions for rasters"""

# Set class as target for resampling
class GroundTruth(object):
    def __init__(self, file):
        self.file = file
        tif = ro.open(file)
        self.labelled_bounds = tif.bounds
        self.height = tif.height
        self.width = tif.width
        self.resolution = tif.transform[0]
        self.array = tif.read()
        
# Normalise pixel values
def normalise(array, nodata):
    """Sets pixels with nodata value to zero then normalises each channel to between 0 and 1"""
    array[array == nodata] = 0
    return (array - array.min(axis=(1, 2))[:, None, None]) / (
        (array.max(axis=(1, 2)) - array.min(axis=(1, 2)))[:, None, None])

# Resample data to target ground truth
def get_training_array(tif, target):
    gt = GroundTruth(target)
    array = tif.read(window=ro.windows.from_bounds(*gt.labelled_bounds + (tif.transform,)),
                     out_shape=(tif.count, gt.height, gt.width),
                     resampling=ro.warp.Resampling.bilinear)
    return normalise(array, tif.nodata)

# Resample tif according to polygon
def resample_tif(file, testPoly, out, verbose=True):
    img = rxr.open_rasterio(file, masked=True).squeeze()
    _, extent = get_extent(testPoly)
    imgCrop = img.rio.clip(extent.geometry.apply(sg.mapping))   
    imgNm = imgCrop.rename('img')
    imgNm.rio.to_raster(out)
    if verbose: print(file+ " read completed.")

# For removing sampling temp files
def del_file_endings(directory, ending):
    for item in os.listdir(directory): 
        if item.endswith(ending): os.remove(item)

          
#----------------------------------------------------#
"""Reprojecting functions"""
# Get file crs
def get_crs(file):
    return str(ro.open(file).crs)

# Convert coordinates between systems
def conv_coords(inFiles, outFiles, crs='EPSG:4326', verbose=True):
    for num, file in enumerate(inFiles):
        with ro.open(file) as src:
            transform, width, height = ro.warp.calculate_default_transform(
                src.crs, crs, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            with ro.open(outFiles[num], 'w', **kwargs) as dst:
                for j in range(1, src.count + 1):
                    ro.warp.reproject(
                        source=ro.band(src, j),
                        destination=ro.band(dst, j),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=crs,
                        resampling=ro.warp.Resampling.nearest)
                if verbose: print(file+" reprojected to "+outFiles[num])
    return outFiles
  

#------------------------------------------#
"""Mosaicing file functions"""
# Rescale tif file
def rescale_tif(tif, scale_factor, mode=ro.warp.Resampling.bilinear):
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
    memory_file = ro.io.MemoryFile().open(**profile)
    memory_file.write(data)
    return memory_file        
        
# Mosaic all tif files in directory according to scaling of target .tif
def mosaic(target, directory):
    file_paths = glob(directory + "/*.tif")
    tif_list = []
    for path in file_paths:
        tif = ro.open(path)
        tif_list.append(rescale_tif(tif, tif.transform[0] / GroundTruth(target).resolution))
    mosaic_tif, mosaic_transform = merge(tif_list)
    profile = tif_list[0].profile
    profile.update({'width': mosaic_tif.shape[-1],
                    'height': mosaic_tif.shape[-2],
                    'transform': mosaic_transform})
    memory_file = ro.io.MemoryFile().open(**profile)
    memory_file.write(mosaic_tif)
    return memory_file