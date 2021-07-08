# Plotting
import sys
import random
import numpy as np
import sklearn as skl
import ipyleaflet as ipl
import matplotlib.pyplot as plt

#----------------------------------------------#
"""ipyleaflet plotting functions"""
# Create ipyleaflet basemap
def create_map(lat, lon, zoom, basemap=ipl.basemaps.OpenStreetMap.Mapnik):
    return ipl.Map(basemap=basemap, center=[lat, lon], zoom=zoom, scroll_wheel_zoom=True)

# Converting gdf columns to GeoData for plotting
def to_geodata(gdf, color, name='Data', fill=0.7):
    plotGdf = ipl.GeoData(geo_dataframe = gdf,
                          style={'color': color, 'radius':2, 'fillColor': color, 'opacity':fill+0.1, 'weight':1.9, 'dashArray':'2', 'fillOpacity':fill},
                          hover_style={'fillColor': 'white' , 'fillOpacity': 0.2},
                          point_style={'radius': 3, 'color': color, 'fillOpacity': 0.8, 'fillColor': color, 'weight': 3},
                          name = name)
    return plotGdf

# Plotting for labels
def plot_labels(gdf, mapName, cn='class', classes=False, colors=False, layer_name='Data'):
    # Extract classes and assign random colors
    classes = gdf[cn].unique() if classes is False else classes
    if colors is False:
        colors = []
        for i in range(len(classes)): colors.append('#%06X' % random.randint(0, 0xFFFFFF))
    leg = {}
    
    # Plot each class with a different color
    for i, cl in enumerate(classes):
        mapName.add_layer(to_geodata(gdf.loc[gdf[cn].str.contains(cl)],colors[i],layer_name))
        leg.update({cl:colors[i]})
        
    # Add legend if forming map for first time
    if not 'l1' in globals(): 
        l1 = ipl.LegendControl(leg, name=cn, position="bottomleft")
        mapName.add_control(l1)
    return mapName

# Draw polygon for labels considered
def draw_polygon(gdf, mapName, stdTest=False, bounds=False):
    bd = gdf.total_bounds if (bounds is False) or (stdTest is False) else bounds # Get bounds from data if not suppplied
    if (bounds is not False) and (stdTest is False): # Non stdTest with supplied bounds only
        sd, wd, nd, ed=0.014, 0.006, 0, 0 # Special adaptation ignoring outlying data points for Beirut 
        locations = [(bd[1]+sd, bd[0]+wd), (bd[1]+sd, bd[2]-ed), (bd[3]-nd, bd[2]-ed),(bd[3]-nd, bd[0]+wd)]
    else: locations = [(bd[1], bd[0]), (bd[1], bd[2]), (bd[3], bd[2]),(bd[3], bd[0])]
    
    # Create polygon for running model
    testPoly = ipl.Polygon(locations = locations, color="yellow", fill_color="yellow",
                           transform=True)
    mapName.add_layer(testPoly)
    return mapName, testPoly
  
# Plotting for building footprints with attached assessments
def plot_assessments(gdf, mapName, cn='decision', classes=['GREEN','YELLOW','RED','TOTAL','LAND'], colors=['green','yellow','red','maroon','cyan'],
                     layer_name='Data', layer_only=False, no_leg=False, fill=0.7, legName=False,names=False,leg_pos='bottomleft'):
    classes = inputs['labels']['decision'].unique() if classes is False else classes 
    leg = {}
    globals()['layer'+layer_name] = ipl.LayerGroup(name = layer_name)
    for i, cl in enumerate(classes):
        try: globals()['layer'+layer_name].add_layer(to_geodata(gdf.loc[gdf[cn].str.contains(cl)],colors[i],layer_name,fill))
        except: globals()['layer'+layer_name].add_layer(to_geodata(gdf.loc[gdf[cn] == cl],colors[i],layer_name,fill))
        if names:   
            namedCls = ['Undamaged', 'Damaged']
            leg.update({namedCls[i]:colors[i]})
        else: leg.update({cl:colors[i]})
            
    if not layer_only:
        mapName.add_layer(globals()['layer'+layer_name])
        if not 'l1' in globals() and no_leg is False: # Add legend if forming map for first time
            l1 = ipl.LegendControl(leg, name=cn if legName is False else legName, position=leg_pos)
            mapName.add_control(l1)
        return mapName
    else: return globals()['layer'+layer_name]
    
def split_contours(segs, kinds=None):
    """takes a list of polygons and vertex kinds and separates disconnected vertices into separate lists.
    The input arrays can be derived from the allsegs and allkinds atributes of the result of a matplotlib
    contour or contourf call. They correspond to the contours of one contour level.
    
    Example:
    cs = plt.contourf(x, y, z)
    allsegs = cs.allsegs
    allkinds = cs.allkinds
    for i, segs in enumerate(allsegs):
        kinds = None if allkinds is None else allkinds[i]
        new_segs = split_contours(segs, kinds)
        # do something with new_segs
        
    More information:
    https://matplotlib.org/3.3.3/_modules/matplotlib/contour.html#ClabelText
    https://matplotlib.org/3.1.0/api/path_api.html#matplotlib.path.Path
    """
    if kinds is None:
        return segs    # nothing to be done
    # search for kind=79 as this marks the end of one polygon segment
    # Notes: 
    # 1. we ignore the different polygon styles of matplotlib Path here and only
    # look for polygon segments.
    # 2. the Path documentation recommends to use iter_segments instead of direct
    # access to vertices and node types. However, since the ipyleaflet Polygon expects
    # a complete polygon and not individual segments, this cannot be used here
    # (it may be helpful to clean polygons before passing them into ipyleaflet's Polygon,
    # but so far I don't see a necessity to do so)
    new_segs = []
    for i, seg in enumerate(segs):
        segkinds = kinds[i]
        boundaries = [0] + list(np.nonzero(segkinds == 79)[0])
        for b in range(len(boundaries)-1):
            new_segs.append(seg[boundaries[b]+(1 if b>0 else 0):boundaries[b+1]])
    return new_segs

#-------------------------------------------#
""" Matplotlib plotting functions"""
# Create matplotlib subplots
def create_subplots(rows, cols, figsize, font=False):
    return plt.subplots(rows, cols, figsize=figsize)

# Display matplotlib plot
def show_plot(): plt.show()

# Save matplotlib plot
def save_plot(fig, filename): fig.savefig(filename)
    
# Display static matplotlib ground truth
def static_label_plot(labelarr):
    # Plot numpy array onto map
    fig, ax = create_subplots(1,1,[10,4])
    a = ax.imshow(labelarr[0],cmap='RdYlBu')
    ax.set_title('Ground Truth',fontsize=13), ax.set_xlabel('x pixels',fontsize=12), ax.set_ylabel('y pixels',fontsize=12)
    ac = fig.colorbar(a,ax=ax)
    ac.set_label('Label Class',fontsize=12)
    show_plot()

#-----------------------------------------#
"""Evaluation metrics plots"""
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
  
def cross_entropy_metrics(axs, y_true, y_pred, classes, dmgThresh=0.5, initBelief=0.5):
    try: ax = axs[1]
    except: ax = axs
    try: 
        int(classes[0]), int(classes[1])
        label1, label2 = 'True class '+str(classes[0]), 'True class '+str(classes[1])
    except: label1, label2 = 'True '+str(classes[0]), 'True '+str(classes[1])
    p1 = ax.hist(y_pred[(np.array(1-y_true)*y_pred).nonzero()[0]], range = [0,1], bins = 100, label = label1, color = 'g', alpha = 0.5) 
    if len(classes) > 1:
        p2 = ax.hist(y_pred[(np.array(y_true)*y_pred).nonzero()[0]], range = [0,1], bins = 100, label =  label2, color = 'r', alpha = 0.5)
    ax.axvline(x=initBelief, color='b',linestyle='--', linewidth=1, label='Initial probability')
    log_loss = skl.metrics.log_loss(y_true, y_pred, labels=[0,1])
    ax.set_title('Belief Propagation\nCross-entropy loss: {:.3f}'.format(log_loss),size=14)
    ax.legend(loc='best',fontsize=12), 
    try: 
        int(classes[0]), int(classes[1])
        ax.set_xlabel('Class '+str(classes[1])+' Probability',fontsize=12)
        ax.text(dmgThresh/2, 0.6, 'Class '+str(classes[0])+'\n Prediction', ha='center', va='center', transform=ax.transAxes,fontsize=12)
        ax.text(dmgThresh+(1-dmgThresh)/2, 0.6, 'Class '+str(classes[1])+'\n Prediction', ha='center', va='center', transform=ax.transAxes,fontsize=12)
    except:
        ax.set_xlabel(str(classes[1])+' Probability',fontsize=12)
        ax.text(dmgThresh/2, 0.6, str(classes[0])+'\n Prediction', ha='center', va='center', transform=ax.transAxes)
        ax.text(dmgThresh+(1-dmgThresh)/2, 0.6, str(classes[1])+'\n Prediction', ha='center', va='center', transform=ax.transAxes)
    ax.set_ylabel('Number of predictions',fontsize=12)
    return axs, log_loss
  
# Cross entropy for multi-class with box plots
def cross_entropy_multiclass(ax, y_true, y_pred, classes):
    a, log_loss = [],0
    for i, val in enumerate(classes):
        a.append((y_pred[:,i].reshape(-1,1)[np.array(y_true)==val]).tolist())
    for i in range(len(classes)):
        if len(a[i]) > 0: log_loss += len(a[i])*skl.metrics.log_loss(np.ones([len(a[i]),1]), a[i], labels=[0,1])
    log_loss = log_loss/len(y_pred)
    ax.set_title('Belief Propagation\nCross-entropy loss: {:.3f}'.format(log_loss),size=14)
    ax.set_xlabel('Classes',fontsize=12), ax.set_ylabel('Probability',fontsize=12)
    try: ax.boxplot(np.array(a), labels=classes)
    except: ax.boxplot(np.array(a)[:,:,0].transpose(), labels=classes)
    ax.hlines(1/y_pred.shape[1],1,len(classes), colors='r', linestyles='dashed', label='Prior belief')
    ax.legend(loc='best',fontsize=12)
    return ax