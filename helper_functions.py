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

def create_map(lat, lon, zoom):
  return ipl.Map(basemap=ipl.basemaps.OpenStreetMap.Mapnik, center=[lat, lon], zoom=zoom, scroll_wheel_zoom=True)

def create_subplots(rows, cols, figsize):
  return plt.subplots(rows, cols, figsize=figsize)

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
#   mapName.add_layer(to_geodata(gdf.loc[gdf['decision'] == 'GREEN (inspected) أخضر (تم دراسته)'],'green'))
#   mapName.add_layer(to_geodata(gdf.loc[gdf['decision'] == 'YELLOW (restricted use) أصفر (لا يصلح للسكن)'],'yellow'))
#   mapName.add_layer(to_geodata(gdf.loc[gdf['decision'] == 'RED (unsafe/evacuate) أحمر (غير آمن/للاخلاء)ء'],'red'))

  if not 'l1' in globals(): # Add legend if forming map for first time
      l1 = ipl.LegendControl({"No Restrictions":"#008000", "Restricted Use":"#FFFF00", "Unsafe/Evacuated":"#FF0000", "No Decision":"#0000FF"}, name="Decision", position="bottomleft")
      mapName.add_control(l1)
  return mapName
  
def draw_polygon(gdf, mapName, stdTest=False):
  bd = gdf.total_bounds
  testPoly = ipl.Polygon(locations = [(bd[1]+0.012, bd[0]), (bd[1]+0.012, bd[2]-0.01), (bd[3], bd[2]-0.01),(bd[3], bd[0])], color="yellow", fill_color="yellow", transform=False if stdTest else True)

  mapName.add_layer(testPoly)
  return mapName, testPoly

## Import interferogram and format into dataframe
def ifg_to_df(ifgFile, polygon):
  # Import interferogram
  wholeIfg = rxr.open_rasterio(ifgFile, masked=True).squeeze()
  # Crop ifg
  try: poly = sg.Polygon([[p['lng'], p['lat']] for p in polygon.locations[0]])
  except: poly = sg.Polygon([[p[1],p[0]] for p in polygon.locations])
  extent = gpd.GeoSeries([poly])
  croppedIfg = wholeIfg.rio.clip(extent.geometry.apply(sg.mapping), extent.crs)
  # Convert to df
  named = croppedIfg.rename('ifg')
  return named.to_dataframe().dropna(subset=['ifg']), poly

def init_beliefs(df, columns, initBeliefs=[0.5,0.5]):
  coords = np.concatenate(np.array(df.axes[0]))
  return gpd.GeoDataFrame(pd.DataFrame(np.concatenate((np.ones([len(df),len(columns)-1])*initBeliefs, np.array(df[columns[-1]]).reshape(-1,1)), axis=1), columns = columns),
                          geometry=gpd.points_from_xy(coords[1::2], coords[0::2]),
                          crs={'init': 'epsg:4326'})
                             
def train_test_split(gdf, poly, column, testSplit=0.3, randomState=1, shuffle=True):
  try: return skms.train_test_split(gdf[['geometry',column]][gdf.within(poly)], 
                                    gdf[column][gdf.within(poly)],
                                    test_size=testSplit, 
                                    random_state=randomState, 
                                    shuffle=shuffle, 
                                    stratify = gdf[column][gdf.within(poly)])
  except ValueError:
    print('Train/Test set stratification not possible due to less than 2 members from smallest class.')
    return skms.train_test_split(gdf[['geometry',column]][gdf.within(poly)], 
                                    gdf[column][gdf.within(poly)],
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

def create_edges(nodes, adjacent=True, geo_neighbors=4, values=False,neighbours=False):
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
  if values:
    for i in range(len(values)):
      edges = edges + np.ndarray.tolist(np.array(kneighbors_graph(np.array(nodes[values[i]]).reshape(-1,1),
                                                                  neighbours[i], 
                                                                  mode='connectivity', 
                                                                  include_self=False).nonzero()).reshape(2,-1).transpose())
  return np.array(edges)

def get_labels(init, X_test, beliefs, values, column):
  
  y_true = gpd.sjoin(init, X_test, how='left', op='within').dropna(subset=[column]).decision.str.split(' ').str[0].map(values)
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

def cross_entropy_metrics(axs, y_true, y_pred, classes, dmgThresh=0.5, initBelief=0.5):
    p1 = axs[1].hist(y_pred[(np.array(1-y_true)*y_pred).nonzero()[0]], range = [0,1], bins = 100, label = 'True '+classes[0], color = 'g', alpha = 0.5)
    if len(classes) > 1:
        p2 = axs[1].hist(y_pred[(np.array(y_true)*y_pred).nonzero()[0]], range = [0,1], bins = 100, label = 'True '+classes[1], color = 'r', alpha = 0.8)
    axs[1].axvline(x=dmgThresh, color='k',linestyle='--', linewidth=1, label='Damage Threshold')
    axs[1].axvline(x=initBelief, color='b',linestyle='--', linewidth=1, label='Initial probability')
    log_loss = skl.metrics.log_loss(y_true, y_pred, labels=[0,1])
    axs[1].set_title('Cross-Entropy loss: {}'.format(log_loss))
    axs[1].legend(loc='upper right'), axs[1].set_xlabel('Damage Probability'), axs[1].set_ylabel('Number of predictions')
    axs[1].text(dmgThresh/2, 0.6,'Undamaged\n Prediction', ha='center', va='center', transform=axs[1].transAxes)
    axs[1].text(dmgThresh+(1-dmgThresh)/2, 0.6,'Damaged\n Prediction', ha='center', va='center', transform=axs[1].transAxes)
    return axs, log_loss
  
def show_plot(): plt.show()
  
def save_plot(fig, filename): fig.savefig(filename)

def belief_plot(nodes, ax, column, normalise = False):
    if normalise: column = skl.preprocessing.normalize(column, norm='l1')[:,1]
    return nodes.plot(ax=ax, column=column, cmap='RdYlGn_r', vmin=0,vmax=1)  
  
def cropped_ifg(ifgFile,polygon):
    wholeIfg = rxr.open_rasterio(ifgFile, masked=True).squeeze()
    # Crop ifg
    try: poly = sg.Polygon([[p['lng'], p['lat']] for p in polygon.locations[0]])
    except: poly = sg.Polygon([[p[1],p[0]] for p in polygon.locations])
    extent = gpd.GeoSeries([poly])
    return wholeIfg.rio.clip(extent.geometry.apply(sg.mapping), extent.crs)