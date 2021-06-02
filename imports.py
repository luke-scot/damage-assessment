import rasterio
import ground_truth
import numpy as np
import pandas as pd
import rioxarray as rxr
import geopandas as gpd
import helper_functions as hf
import shapely.geometry as sg
from rasterio.io import MemoryFile
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

#----------------------------------------------------#
"""Label Imports"""
def shape_to_gdf(file, splitString = False, cn='decision', crs=False):
    labels = gpd.read_file(file)
    if crs: labels= labels.to_crs({'init': crs})
    if splitString: labels[cn] = labels[cn].str.split(' ').str[0]
    return labels
  
#-----------------------------------------------------#
"""Beirut Data imports"""
# Import OSM building footprints from .geojson file and return geodataframe
def import_OSM_fps(buildingGeojson):
  buildings = gpd.read_file(buildingGeojson)
  return gpd.GeoDataFrame(buildings[['id','building','name']], geometry=buildings.geometry)

# Extract coordinates from columns - used for GeoPal Data
def get_coords(locations, mapPoints):
    lats, lons = np.zeros([len(locations), 1]),  np.zeros([len(locations), 1])
    for i in range(len(locations)):
        loc = locations[i]
        if type(loc) is float or (type(loc) is str and loc[0].isalpha()):
            mp = mapPoints[i]
            if type(mp) is str and mp[0].isdigit():
                try: lats[i], lons[i] = mp.split(' ')[0], mp.split(' ')[1]
                except: lats[i], lons[i] = mp.split(',')[0], mp.split(',')[1] # Deal with rogue commas instead of space
        else: lats[i], lons[i] = loc.split(' ')[0], loc.split(' ')[1]
    return lats, lons
  
# Import GeoPal data, extract locations from columns and return geodataframe of located data
def import_located_geopal_data(geopalCsv):
  allData = pd.read_csv(geopalCsv)
  # Extract locations from joint column in database
  locations, mapPoints = allData['get location - الموقع_w_2049198'], allData['point on map - الموقع على الخريطة_w_2049199']
  lats, lons = get_coords(locations, mapPoints)

  # Extract columns of useful data
  data = pd.DataFrame({
      'id': allData['Job ID'],
      'area': allData['plot area - المنطقة العقارية_w_2049201'],
      'damage': allData['structural damage level - مستوى الضرر الأنشائي للمبنى_w_2049205'],
      'floors': allData['number of floors - عدد الطوابق_w_2049208'],
      'units': allData['number of units - عدد الشقق_w_2049209'],
      'use': allData['building use - وجهة الاستعمال للمبنى_w_2049210'],
      'photos': allData['take pictures - التقاط صور_w_2049222'],
      'decision': allData['decision - القرار_w_2049224']    
  })

  # Create geodatabase merging locations with useful data
  assessments = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(lons, lats),crs={'init': 'epsg:4326'})

  # Filter for non located values
  return assessments[assessments.geometry.x != 0]

# Append additional data from GeoPal
def append_geopal_data(orig, filePath, decCol = 'decision', extractCoords = False, coordCols = ['get location - الموقع_w_2048240', 'point on map - الموقع على الخريطة_w_2048241']):
    extraData = pd.read_csv(filePath)
    if extractCoords: 
        lats, lons = get_coords(extraData[coordCols[0]], extraData[coordCols[1]])
    else:
        lats, lons = extraData['Lat'], extraData['Lon']
    new = orig.append(gpd.GeoDataFrame({'decision':extraData[decCol]}, geometry=gpd.points_from_xy(lons, lats), crs={'init': 'epsg:4326'}), ignore_index=True)
    return new[new.geometry.x != 0]

#-----------------------------------------------------#
"""Image import functions"""
# Import tif image and format into dataframe
def image_to_df(imgFile, label='label', poly=False):
  # Import image
  img = rxr.open_rasterio(imgFile, masked=True).squeeze()
  # Crop image if polygon supplied
  if poly:
      polygon = poly
      try: poly = sg.Polygon([[p['lng'], p['lat']] for p in polygon.locations[0]])
      except: poly = sg.Polygon([[p[1],p[0]] for p in polygon.locations])
      extent = gpd.GeoSeries([poly])
      img = img.rio.clip(extent.geometry.apply(sg.mapping), extent.crs)
  # Convert to df
  named = img.rename(label)
  return named.to_dataframe().dropna(subset=[label]), poly

#-----------------------------------------------------#
"""Raster import functions"""
# Convert image raster to array
def raster_to_array(file, crop=False):
    img = rasterio.open(file)
    return get_training_array(img) if crop else img.read()

# Get training data from ground_truth class (ground_truth.py must be in folder)
def get_training_array(tif):
    array = tif.read(window=from_bounds(*ground_truth.data.labelled_bounds + (tif.transform,)),
                     out_shape=(tif.count, ground_truth.data.height, ground_truth.data.width),
                     resampling=Resampling.bilinear)
    return normalise(array, tif.nodata)

# Normalise pixel values
def normalise(array, nodata):
    """Sets pixels with nodata value to zero then normalises each channel to between 0 and 1"""
    array[array == nodata] = 0
    return (array - array.min(axis=(1, 2))[:, None, None]) / (
        (array.max(axis=(1, 2)) - array.min(axis=(1, 2)))[:, None, None])

# Resample tif to equal resolution of ground truth  
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

# Overall function taking a raster input and outputting pandas dataframe
def raster_to_df(file, value='class', multidims = False, crop = False):
    arr = raster_to_array(file, crop)
    if multidims:
        df = pd.DataFrame(arr.reshape(-1,len(arr)))
        df['x'], df['y'] = np.tile(np.arange(arr.shape[2]),arr.shape[1]), np.repeat(range(arr.shape[1]),arr.shape[2])
        df = df.set_index(['y','x'])
    else: df = pd.DataFrame(arr[0]).stack().rename_axis(['y', 'x']).reset_index(name=value).set_index(['y','x'])
    return df, arr
  
#---------------------------------------------------------#
"""High-Resolution Imports"""
def img_to_gdf(file, poly=False, crs=False, label='img', columns=False, crsPoly='epsg:4326'):
    # High-Resolution Imagery import
    img = rxr.open_rasterio(file, masked=True).squeeze()
    # Crop image if polygon supplied
    if poly:
        extent = hf.get_extent(poly, crsPoly=crsPoly, crs=crs)
        img = img.rio.clip(extent.geometry.apply(sg.mapping))   
    named = img.rename('img')
    
    # Convert to geodataframe
    xm, ym = np.meshgrid(np.array(named.coords['x']), np.array(named.coords['y']))
    mi = pd.MultiIndex.from_arrays([ym.flatten(),xm.flatten()],names=('y','x'))

    # Convert to geodataframe with lat/long
    df = pd.DataFrame(named.data.reshape(3,-1).transpose(), index=mi)
    columns = df.columns if columns is False else columns
    gdf = hf.df_to_gdf(df,columns,crs=crs, reIndex=True).to_crs({'init':crsPoly})
    return gdf