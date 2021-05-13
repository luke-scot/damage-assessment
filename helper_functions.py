import geopandas as gpd
import pandas as pd
import numpy as np
import ipyleaflet as ipl
import rasterio as ro
import rioxarray as rxr
import sklearn as skl
import shapely.geometry as sg
import matplotlib.pyplot as plt
import sklearn.model_selection as skms
from sklearn.neighbors import BallTree, kneighbors_graph
from matplotlib.colors import LogNorm
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def create_map(lat, lon, zoom):
  return ipl.Map(basemap=ipl.basemaps.OpenStreetMap.Mapnik, center=[lat, lon], zoom=zoom, scroll_wheel_zoom=True)

def create_subplots(rows, cols, figsize):
  return plt.subplots(rows, cols, figsize=figsize)

def plot_image(file, ax, fig=False, title=False, log=None):
    im = Image.open(file)
    if log: log=LogNorm()
    p = ax.imshow(np.array(im), norm=log)
    if fig: fig.colorbar(p, ax=ax)
    if title: ax.set_title(title)
    return p

# Import OSM building footprints from .geojson file and return geodataframe
def import_OSM_fps(buildingGeojson):
  buildings = gpd.read_file(buildingGeojson)
  return gpd.GeoDataFrame(buildings[['id','building','name']], geometry=buildings.geometry)

# Extract coordinates from columns
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

# Join geodataframes
def join_gdfs(gdf1, gdf2, column):
  return gpd.sjoin(gdf1, gdf2, how="left", op='contains').dropna(subset=[column])

# Converting gdf columns to GeoData for plotting
def to_geodata(gdf, color):
    plotGdf = ipl.GeoData(geo_dataframe = gdf,
                          style={'color': color, 'radius':2, 'fillColor': color, 'opacity':0.9, 'weight':1.9, 'dashArray':'2', 'fillOpacity':0.7},
                          hover_style={'fillColor': 'white' , 'fillOpacity': 0.2},
                          point_style={'radius': 3, 'color': color, 'fillOpacity': 0.8, 'fillColor': color, 'weight': 3},
                          name = 'Images')
    return plotGdf

# Plotting for building footprints with attached assessments
def plot_assessments(gdf, mapName):
  mapName.add_layer(to_geodata(gdf.loc[gdf['decision'].str.contains('GREEN')],'green'))
  mapName.add_layer(to_geodata(gdf.loc[gdf['decision'].str.contains('YELLOW')],'yellow'))
  mapName.add_layer(to_geodata(gdf.loc[gdf['decision'].str.contains('RED')],'red'))

  if not 'l1' in globals(): # Add legend if forming map for first time
      l1 = ipl.LegendControl({"No Restrictions":"#008000", "Restricted Use":"#FFFF00", "Unsafe/Evacuated":"#FF0000", "No Decision":"#0000FF"}, name="Decision", position="bottomleft")
      mapName.add_control(l1)
  return mapName
  
def draw_polygon(gdf, mapName, stdTest=False):
  bd = gdf.total_bounds
  testPoly = ipl.Polygon(locations = [(bd[1]+0.012, bd[0]), (bd[1]+0.012, bd[2]-0.01), (bd[3], bd[2]-0.01),(bd[3], bd[0])], color="yellow", fill_color="yellow", transform=False if stdTest else True)

  mapName.add_layer(testPoly)
  return mapName, testPoly

## Import tif image and format into dataframe
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


def init_beliefs(df, columns, initBeliefs=[0.5,0.5], crs='epsg:4326'):
    for i, val in enumerate(columns): df[val] = np.ones([len(df)])*initBeliefs[i]
    return df_to_gdf(df,df.columns,crs=crs,reIndex=True) 
# def init_beliefs(df, columns, initBeliefs=[0.5,0.5], crs='epsg:4326'):
#   coords = np.concatenate(np.array(df.axes[0]))
#   return gpd.GeoDataFrame(pd.DataFrame(np.concatenate((np.ones([len(df),len(columns)-1])*initBeliefs, np.array(df[columns[-1]]).reshape(-1,1)), axis=1), columns = columns),
#                           geometry=gpd.points_from_xy(coords[1::2], coords[0::2]),
#                           crs={'init': crs})

# Pandas dataframe to formatted geodataframe
def df_to_gdf(df, columns, crs='epsg:4326', reIndex=False):
  coords = np.concatenate(np.array(df.axes[0]))
  gdf = gpd.GeoDataFrame(df[columns[:]], columns = columns, geometry=gpd.points_from_xy(coords[1::2], coords[0::2]),crs={'init': crs})
  if reIndex:
    gdf = gdf.reset_index()
    del gdf['x']
    del gdf['y']
  return gdf
                             
def train_test_split(gdf, column, poly=False, testSplit=0.3, randomState=1, shuffle=True):
    try: 
        if poly: return skms.train_test_split(gdf[['geometry',column]][gdf.within(poly)], 
                                    gdf[column][gdf.within(poly)],
                                    test_size=testSplit, 
                                    random_state=randomState, 
                                    shuffle=shuffle, 
                                    stratify = gdf[column][gdf.within(poly)])
        else: return skms.train_test_split(gdf[['geometry',column]], 
                                    gdf[column],
                                    test_size=testSplit, 
                                    random_state=randomState, 
                                    shuffle=shuffle, 
                                    stratify = gdf[column])
    except ValueError:
        print('Train/Test set stratification not possible due to less than 2 members from smallest class.')
        if poly: return skms.train_test_split(gdf[['geometry',column]][gdf.within(poly)], 
                                    gdf[column][gdf.within(poly)],
                                    test_size=testSplit, 
                                    random_state=randomState, 
                                    shuffle=shuffle)
        else: return skms.train_test_split(gdf[['geometry',column]], 
                                    gdf[column],
                                    test_size=testSplit, 
                                    random_state=randomState, 
                                    shuffle=shuffle)

def create_nodes(init, X_train):
  nodes = gpd.sjoin(init, X_train, how='left', op='within')
  return nodes[~nodes.index.duplicated(keep='first')]
                          
def prior_beliefs(nodes, values, beliefs, column, beliefColumns):
  for i in range(len(beliefs)):
    nodes.loc[nodes[column] == list(values.keys())[i], beliefColumns] = 1-beliefs[i], beliefs[i]
  return np.array(nodes[beliefColumns])

def create_edges(nodes, adjacent=True, geo_neighbors=4, values=False, neighbours=False):
    edges = []
    # Create edges between geographically adjacent nodes
    if adjacent:
        points = np.array([nodes.geometry.x,nodes.geometry.y]).transpose()
        tree = BallTree(points, leaf_size=15, metric='haversine')
        _, ind = tree.query(points, k=geo_neighbors+1)
        for i in np.arange(1,ind.shape[1]):
            edges = edges + np.ndarray.tolist(np.array([ind[:,0],ind[:,i]]).transpose())
        edges = np.array(edges)
        edges.sort(axis=1)
        edges = np.ndarray.tolist(np.unique(edges, axis=0))

    # Create edges between most similar phase change pixels
    if values is not False:
        edges = edges + np.ndarray.tolist(np.array(kneighbors_graph(np.array(nodes[values]),2,mode='connectivity',include_self=False).nonzero()).reshape(2,-1).transpose())
    return np.array(edges)
#   if values:
#     for i in range(len(values)):
#       edges = edges + np.ndarray.tolist(np.array(kneighbors_graph(np.array(nodes[values[i]]).reshape(-1,1))

def get_labels(init, X_test, beliefs, values, column, splitString=False):
    if splitString: y_true = gpd.sjoin(init, X_test, how='left', op='within').dropna(subset=[column]).decision.str.split(' ').str[0].map(values)
    else: y_true = gpd.sjoin(init, X_test, how='left', op='within').dropna(subset=[column])[column].map(values)
    y_pred = skl.preprocessing.normalize(beliefs[y_true.index], norm='l1')[:,1]
    return y_true, y_pred

def class_metrics(y_true, y_pred, targets, threshold=0.5):
  yp_clf = skl.preprocessing.binarize(y_pred.reshape(-1, 1), threshold=threshold)
  classes = targets[:len(np.unique(np.append(yp_clf, np.array(y_true))))]
  print(skl.metrics.classification_report(y_true, yp_clf, target_names=classes, zero_division=0))
  return yp_clf, classes

def confusion_matrix(axs, y_true, yp_clf, classes):
  conf = skl.metrics.confusion_matrix(y_true, yp_clf)
  axs[0].imshow(conf, interpolation='nearest')
  axs[0].set_xticks(range(len(classes))), axs[0].set_xticklabels(classes), axs[0].set_yticks(range(len(classes))), axs[0].set_yticklabels(classes)
  axs[0].set_xlabel('Predicted Class'), axs[0].set_ylabel('True Class'), axs[0].set_title('Confusion Matrix')
  for i in range(len(classes)): 
      for j in range(len(classes)): text = axs[0].text(j, i, conf[i, j], ha="center", va="center", color="r")
  return axs

# Evaluate the cross entropy metrics and plot histogram of individual beliefs
def cross_entropy_metrics(axs, y_true, y_pred, classes, dmgThresh=0.5, initBelief=0.5):
    p1 = axs[1].hist(y_pred[(np.array(1-y_true)*y_pred).nonzero()[0]], range = [0,1], bins = 100, label = 'True '+classes[0], color = 'g', alpha = 0.5)
    if len(classes) > 1:
        p2 = axs[1].hist(y_pred[(np.array(y_true)*y_pred).nonzero()[0]], range = [0,1], bins = 100, label = 'True '+classes[1], color = 'r', alpha = 0.5)
    axs[1].axvline(x=dmgThresh, color='k',linestyle='--', linewidth=1, label='Damage Threshold')
    axs[1].axvline(x=initBelief, color='b',linestyle='--', linewidth=1, label='Initial probability')
    log_loss = skl.metrics.log_loss(y_true, y_pred, labels=[0,1])
    axs[1].set_title('Cross-Entropy loss: {}'.format(log_loss))
    axs[1].legend(loc='upper right'), axs[1].set_xlabel('Damage Probability'), axs[1].set_ylabel('Number of predictions')
    axs[1].text(dmgThresh/2, 0.6, classes[0]+'\n Prediction', ha='center', va='center', transform=axs[1].transAxes)
    axs[1].text(dmgThresh+(1-dmgThresh)/2, 0.6, classes[1]+'\n Prediction', ha='center', va='center', transform=axs[1].transAxes)
    return axs, log_loss

# Display matplotlib plot
def show_plot(): plt.show()
  
# Save matplotlib plot
def save_plot(fig, filename): fig.savefig(filename)

# Plot resulting beliefs from NetConf
def belief_plot(nodes, ax, column, normalise = False):
    if normalise: column = skl.preprocessing.normalize(column, norm='l1')[:,1]
    return nodes.plot(ax=ax, column=column, cmap='RdYlGn_r', vmin=0,vmax=1)  

# Crop an interferogram
def cropped_ifg(ifgFile,polygon):
    wholeIfg = rxr.open_rasterio(ifgFile, masked=True).squeeze()
    # Crop ifg
    try: poly = sg.Polygon([[p['lng'], p['lat']] for p in polygon.locations[0]])
    except: poly = sg.Polygon([[p[1],p[0]] for p in polygon.locations])
    extent = gpd.GeoSeries([poly])
    return wholeIfg.rio.clip(extent.geometry.apply(sg.mapping), extent.crs)

# Create an ipyleaflet polygon from properties
def create_ipl_polygon(locations, color="yellow", fill_color="yellow", transform=False):
    return ipl.Polygon(locations = locations, color=color, fill_color=fill_color, transform=transform)
  
# Group classes of labels according to classes matrix (1 row per class - [min val, max val])
def group_classes(labels, classes, zeroNan=False, intervals = False):
    for i in range(len(classes)):
        if zeroNan: np.where((labels == 0), np.nan, labels)
        if intervals: labels = np.where((labels >= classes[i][0]) & (labels <= classes[i][1]), i, labels)
        else: labels = np.where((pd.Series(labels).isin(classes[i])), i, labels)
    return labels 

def run_PCA(X, n=2):
  pca = PCA(n_components=2)
  return pca.fit(X)

def run_kmeans(X, clusters=2, rs=0):
    return KMeans(n_clusters=clusters, random_state=rs).fit(X)