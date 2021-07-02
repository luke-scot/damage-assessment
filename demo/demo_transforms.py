import os
import random
import importlib
import ground_truth
import numpy as np
import pandas as pd
import rasterio as ro
import rioxarray as rxr
import geopandas as gpd
import helper_functions as hf
import shapely.geometry as sg
from rasterio.io import MemoryFile
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling

#----------------------------------------------------#
"""Label Imports"""
def shape_to_gdf(file, splitString = False, cn='class', crs=False):
    labels = gpd.read_file(file)
    if crs: labels=labels.to_crs({'init': crs})
    if cn not in labels.columns: raise ValueError(cn+" column not found in labels dataframe.")
    if splitString: labels[cn] = labels[cn].str.split(' ').str[0]
    return labels
