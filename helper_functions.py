import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.geometry as sg
import sklearn as skl
import sklearn.model_selection as skms
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import BallTree, kneighbors_graph

#---------------------------------------------------------#
"""Dataframe manipulation functions"""
# Pandas dataframe to formatted geodataframe
def df_to_gdf(df, columns, crs='epsg:4326', reIndex=False, invCoords=False):
  coords = np.concatenate(np.array(df.axes[0]))
  if invCoords: a,b = 0,1
  else: a,b =1,0
  gdf = gpd.GeoDataFrame(df[columns[:]], columns = columns, geometry=gpd.points_from_xy(coords[a::2], coords[b::2]),crs=crs)
  if reIndex:
    gdf = gdf.reset_index()
    del gdf['x']
    del gdf['y']
  return gdf

# Concatenate dataframes
def concat_dfs(dfs, axis=1):
    return pd.concat(dfs, axis=axis)
  
# Join geodataframes
def join_gdfs(gdf1, gdf2, column):
  return gpd.sjoin(gdf1, gdf2, how="left", op='contains').dropna(subset=[column])

def flatten_list(listIn):
  return list(np.array(listIn).flatten())

def subtract_gdfs(gdf1, gdf2, geometry='geometry'):
  gdf = gdf1-gdf2
  gdf[geometry]=gdf2[geometry]
  gdf.crs = gdf2.crs
  return gdf

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
      
#--------------------------------------------------------------#
"""Class clustering functions"""
# Principal Component Analysis
def run_PCA(X, n=2):
  pca = PCA(n_components=2)
  return pca.fit(X)

# K-means clustering implementation
def run_kmeans(X, clusters=2, rs=0):
    return KMeans(n_clusters=clusters, random_state=rs).fit(X)

# Pre-process hyperspectral data
def run_cluster(X, labels, meanCluster = True, nClasses = 2):
    # Run clustering on all data
    if meanCluster:
        a = pd.DataFrame(X)
        a['label']=labels.values.reshape(-1,1)
        kmeans = run_kmeans(a.groupby(['label']).mean().values, clusters=nClasses)
        classes = kmeans.labels_
        initLabels = list(np.array(a.groupby(['label']).mean().axes[0]))
    else:
        k1 = run_kmeans(X, clusters=nClasses)
        a = pd.DataFrame(np.concatenate((k1.labels_.reshape(1,-1), labels.values.reshape(1,-1))).transpose(),columns=['group','label'])
        kmeans = a.groupby(['label','group']).size()
        b = np.zeros([len(labels.unique())+1,nClasses])
        for i, val in enumerate(kmeans.axes[0]): b[i] = kmeans[i] 
        classes = run_kmeans(b[1:],nClasses).labels_
    return kmeans, classes, initLabels

# Get default classes from original labels
def default_classes(labels):
  return np.sort(labels.unique())
  
#--------------------------------------------------------#
"""Node and Edge creation functions"""
# Initialise beliefs for each class
def init_beliefs(df, classes=2, columns=False, initBeliefs=False, crs='epsg:4326'):
    if columns is False: columns = ['cl'+str(s) for s in range(classes)] # Default column headers 
    if initBeliefs is False: initBeliefs = np.ones(len(columns))*(1/len(columns))
    for i, val in enumerate(columns): df[val] = np.ones([len(df)])*initBeliefs[i]
    return df_to_gdf(df,df.columns,crs=crs,reIndex=True) if type(df) is not gpd.geodataframe.GeoDataFrame else df
  
# Group classes of labels according to classes matrix (1 row per class - [min val, max val])
def group_classes(labels, classes, zeroNan=False, intervals = False):
    for i in range(len(classes)):
        if zeroNan: np.where((labels == 0), np.nan, labels)
        if intervals: labels = np.where((labels >= classes[i][0]) & (labels <= classes[i][1]), i, labels)
        else: labels = np.where((pd.Series(labels).isin(classes[i])), i, labels)
    return labels 

# Split data into train and test sets
def train_test_split(gdf, column, poly=False, testSplit=0.3, randomState=1, shuffle=True, splitString=False):
    if splitString: gdf[column]=gdf[column].str.split(' ').str[0]
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

# Create graph nodes
def create_nodes(init, X_train):
  nodes = gpd.sjoin(init, X_train, how='left', op='within')
  return nodes[~nodes.index.duplicated(keep='first')]

# Extract prior beliefs from nodes as input to belief propagation
def prior_beliefs(nodes, beliefColumns, classNames=False, beliefs=[0,1], column='class'):
    oneHot = np.eye(len(beliefColumns))*beliefs[1]+np.ones(len(beliefColumns))*beliefs[0]-np.eye(len(beliefColumns))*beliefs[0]
    for i, cl in enumerate(beliefColumns):
        match = i if cl == 'cl'+str(i) or (classNames is not False and len(classNames)==len(beliefColumns)) else cl
        nodes.loc[nodes[column] == match, beliefColumns] = oneHot[i,:]
    return np.array(nodes[beliefColumns])

# Create edges between nodes according to similarity
def create_edges(nodes, adjacent=True, geo_neighbours=4, values=False, neighbours=[2]):
    edges = []
    # Create edges between geographically adjacent nodes
    if adjacent and (geo_neighbours > 0):
        points = np.array([nodes.geometry.x,nodes.geometry.y]).transpose()
        tree = BallTree(points, leaf_size=15, metric='haversine')
        _, ind = tree.query(points, k=geo_neighbours+1)
        for i in np.arange(1,ind.shape[1]):
            edges = edges + np.ndarray.tolist(np.array([ind[:,0],ind[:,i]]).transpose())
        edges = np.array(edges)
        edges.sort(axis=1)
        edges = np.ndarray.tolist(np.unique(edges, axis=0))

    # Create edges between most similar phase change pixels
    if values is not False:
        for i, val in enumerate(values):
          if neighbours[i] > 0:
            edges = edges + np.ndarray.tolist(np.array(kneighbors_graph(np.array(nodes[val]).reshape(-1,len(np.array(val))),neighbours[i],mode='connectivity',include_self=False).nonzero()).reshape(2,-1).transpose())
#           edges = edges + np.ndarray.tolist(np.array(kneighbors_graph(np.array(nodes[val]).reshape(-1,len(np.array([val]))),neighbours[i],mode='connectivity',include_self=False).nonzero()).reshape(2,-1).transpose())
#            edges = edges + np.ndarray.tolist(np.array(kneighbors_graph(np.array(nodes[val]).reshape(-1,len(list(val))),neighbours[i],mode='connectivity',include_self=False).nonzero()).reshape(2,-1).transpose())
    return np.array(edges)
  
#---------------------------------------------#
"""Evaluating metrics functions"""
# Get y_true and y_pred for test set oof nodes
def get_labels(init, X_test, beliefs, column, values = False):
    y_true = gpd.sjoin(init, X_test, how='left', op='within').dropna(subset=[column])[column]
    if values: y_true = y_true.map(values)
    y_pred = skl.preprocessing.normalize(beliefs[y_true.index], norm='l1')
    return np.array(y_true).reshape(-1,1).astype(type(y_true.values[0])), y_pred

# Obtain classification report for classes
def class_metrics(y_true, y_pred, classes, orig):
    yp_clf = np.argmax(y_pred, axis=1)
    d = dict(enumerate(classes))
    pred_clf = [d.get(i) for i in yp_clf]
    true_clf = [d.get(i[0]) for i in y_true] if len(classes) < len(orig) else y_true
    print(skl.metrics.classification_report(true_clf, pred_clf,zero_division=0))#, target_names=classes, zero_division=0))
    return true_clf, pred_clf  