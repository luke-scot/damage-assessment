import numpy as np
import sklearn as skl
import rioxarray as rxr
import geopandas as gpd
import ipyleaflet as ipl
import shapely.geometry as sg
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import LogNorm

#---------------------------------------------#
""" Matplotlib plotting functions"""
# Create matplotlib subplots
def create_subplots(rows, cols, figsize, font=False):
  return plt.subplots(rows, cols, figsize=figsize)

# Display matplotlib plot
def show_plot(): plt.show()
  
# Save matplotlib plot
def save_plot(fig, filename): fig.savefig(filename)
  
#----------------------------------------------#
"""Map plotting functions"""
# Create ipyleaflet basemap
def create_map(lat, lon, zoom, basemap=ipl.basemaps.OpenStreetMap.Mapnik):
    return ipl.Map(basemap=basemap, center=[lat, lon], zoom=zoom, scroll_wheel_zoom=True)
  
# Converting gdf columns to GeoData for plotting
def to_geodata(gdf, color, name='Data'):
    plotGdf = ipl.GeoData(geo_dataframe = gdf,
                          style={'color': color, 'radius':2, 'fillColor': color, 'opacity':0.9, 'weight':1.9, 'dashArray':'2', 'fillOpacity':0.7},
                          hover_style={'fillColor': 'white' , 'fillOpacity': 0.2},
                          point_style={'radius': 3, 'color': color, 'fillOpacity': 0.8, 'fillColor': color, 'weight': 3},
                          name = name)
    return plotGdf

# Plotting for building footprints with attached assessments
def plot_assessments(gdf, mapName, cn='decision', classes=['GREEN','YELLOW','RED','TOTAL','LAND'], colors=['green','yellow','red','maroon','cyan'], layer_name='Data'):
  classes = inputs['labels']['decision'].unique() if classes is False else classes 
  leg = {}
  for i, cl in enumerate(classes):
    mapName.add_layer(to_geodata(gdf.loc[gdf[cn].str.contains(cl)],colors[i],layer_name))
    leg.update({cl:colors[i]})

  if not 'l1' in globals(): # Add legend if forming map for first time
      l1 = ipl.LegendControl(leg, name=cn, position="bottomleft")
      mapName.add_control(l1)
  return mapName

def draw_polygon(gdf, mapName, stdTest=False, sd=0.014, wd=0.006, nd=0, ed=0):
  bd = gdf.total_bounds
  testPoly = ipl.Polygon(locations = [(bd[1]+sd, bd[0]+wd), (bd[1]+sd, bd[2]-ed), (bd[3]-nd, bd[2]-ed),(bd[3]-nd, bd[0]+wd)],
                         color="yellow",
                         fill_color="yellow",
                         transform=False if stdTest else True)
  mapName.add_layer(testPoly)
  return mapName, testPoly

# Create an ipyleaflet polygon from properties
def create_ipl_polygon(locations, color="yellow", fill_color="yellow", transform=False):
    return ipl.Polygon(locations = locations, color=color, fill_color=fill_color, transform=transform)

#--------------------------------------------#
"""Image plotting functions"""
# Plot image using PIL library
def plot_image(file, ax, fig=False, title=False, log=None):
    im = Image.open(file)
    if log: log=LogNorm()
    p = ax.imshow(np.array(im), norm=log)
    if fig: fig.colorbar(p, ax=ax)
    if title: ax.set_title(title)
    return p
  
# Crop an interferogram
def cropped_ifg(ifgFile,polygon):
    wholeIfg = rxr.open_rasterio(ifgFile, masked=True).squeeze()
    # Crop ifg
    try: poly = sg.Polygon([[p['lng'], p['lat']] for p in polygon.locations[0]])
    except: poly = sg.Polygon([[p[1],p[0]] for p in polygon.locations])
    extent = gpd.GeoSeries([poly])
    return wholeIfg.rio.clip(extent.geometry.apply(sg.mapping), extent.crs)

#---------------------------------------------#
"""Belief propagation results plots"""
# Plot resulting beliefs from NetConf
def belief_plot(nodes, ax, column, normalise = False):
    if normalise: column = skl.preprocessing.normalize(column, norm='l1')[:,1]
    return nodes.plot(ax=ax, column=column, cmap='RdYlGn_r', vmin=0,vmax=1)  

# Create confusion matrix for all classes contained in y_true and y_pred
def confusion_matrix(axs, true_clf, pred_clf, classes=None):
    conf = skl.metrics.confusion_matrix(true_clf, pred_clf, classes)
    try: ax = axs[0]
    except: ax = axs
    ax.imshow(conf, interpolation='nearest')
    ax.set_xticks(range(len(classes))), ax.set_xticklabels(classes), ax.set_yticks(range(len(classes))), ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted Class',fontsize=12), ax.set_ylabel('True Class',fontsize=12)
    ax.set_title('Confusion Matrix\n F1: {:.3f}  Acc: {:.3f}   Rec: {:.3f}'.format(skl.metrics.f1_score(true_clf, pred_clf,average='weighted',zero_division=0),
                                                                                                   skl.metrics.accuracy_score(true_clf, pred_clf),
                                                                                                   skl.metrics.recall_score(true_clf, pred_clf,average='weighted',zero_division=0)), size=14)
    for i in range(len(classes)): 
        for j in range(len(classes)): text = ax.text(j, i, conf[i, j], ha="center", va="center", color="r")
    return axs
  
# Evaluate the cross entropy metrics and plot histogram of individual beliefs
def cross_entropy_metrics(axs, y_true, y_pred, classes, dmgThresh=0.5, initBelief=0.5):
    try: ax = axs[1]
    except: ax = axs
    p1 = ax.hist(y_pred[(np.array(1-y_true)*y_pred).nonzero()[0]], range = [0,1], bins = 100, label = 'True '+str(classes[0]) if type(classes[0]) is str else 'True class '+str(classes[0]), color = 'g', alpha = 0.5)
    if len(classes) > 1:
        p2 = ax.hist(y_pred[(np.array(y_true)*y_pred).nonzero()[0]], range = [0,1], bins = 100, label = 'True '+str(classes[1]) if type(classes[1]) is str else  'True class '+str(classes[1]), color = 'r', alpha = 0.5)
#   ax.axvline(x=dmgThresh, color='k',linestyle='--', linewidth=1, label='Classification Threshold')
    ax.axvline(x=initBelief, color='b',linestyle='--', linewidth=1, label='Initial probability')
    log_loss = skl.metrics.log_loss(y_true, y_pred, labels=[0,1])
    ax.set_title('Belief Propagation\nCross-entropy loss: {:.3f}'.format(log_loss),size=14)
    ax.legend(loc='best',fontsize=12), 
    if type(classes[1]) is str: 
        ax.set_xlabel(str(classes[1])+' Probability',fontsize=12)
        ax.text(dmgThresh/2, 0.6, str(classes[0])+'\n Prediction', ha='center', va='center', transform=ax.transAxes)
        ax.text(dmgThresh+(1-dmgThresh)/2, 0.6, str(classes[1])+'\n Prediction', ha='center', va='center', transform=ax.transAxes)
    else: 
        ax.set_xlabel('Class '+str(classes[1])+' Probability',fontsize=12)
        ax.text(dmgThresh/2, 0.6, 'Class '+str(classes[0])+'\n Prediction', ha='center', va='center', transform=ax.transAxes,fontsize=12)
        ax.text(dmgThresh+(1-dmgThresh)/2, 0.6, 'Class '+str(classes[1])+'\n Prediction', ha='center', va='center', transform=ax.transAxes,fontsize=12)
    ax.set_ylabel('Number of predictions',fontsize=12)
    return axs, log_loss
  
# Cross entropy for multi-class with box plots
def cross_entropy_multiclass(ax, y_true, y_pred, classes):
    a, log_loss = [],0
    for i, val in enumerate(classes):
        a.append(y_pred[:,i].reshape(-1,1)[np.array(y_true)==val])

    for i in range(len(classes)):
        if len(a[i]) > 0: log_loss += len(a[i])*skl.metrics.log_loss(np.ones([len(a[i]),1]), a[i], labels=[0,1])
    log_loss = log_loss/len(y_pred)
    
    ax.set_title('Belief Propagation\nCross-entropy loss: {:.3f}'.format(log_loss),size=14)
    ax.set_xlabel('Classes',fontsize=12), ax.set_ylabel('Probability',fontsize=12)
    ax.boxplot(a, labels=classes)
    ax.hlines(1/y_pred.shape[1],1,len(classes), colors='r', linestyles='dashed', label='Prior belief')
    ax.legend(loc='best',fontsize=12)
    return ax