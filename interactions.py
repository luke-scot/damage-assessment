import os
import math
import imageio
import numpy as np
import geopandas as gpd
import ipywidgets as ipw
import ipyleaflet as ipl
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
from branca.colormap import linear
from ipyleaflet import LayersControl
from scipy.interpolate import griddata

import netconf as nc
import demo_plotting as pl
import demo_imports as ip
import demo_transforms as tr

#--------------------------------------------#
"""Input Functions"""
# Ask for default assessment to be used
def get_defaults():
    layout = {'width': 'max-content'}
    options = ['Beirut damage assessment', 'Houston land classification', 'None']
    bxDefaults = ipw.Box([ipw.Label(value='Please select default inputs: '),
                          ipw.Dropdown(options=options,value=options[0],disabled=False,layout=layout)])
    display(bxDefaults)
    v={'layout':layout,'options':options,'bxDefaults':bxDefaults}
    return v
  
# Ask for input parameters
def input_parameters(v):
    # Retrieve variables from default
    for i in v.keys(): globals()[i] = v[i]
    
    # Standard defaults - change here to add more or modify existing
    # Beirut
    beirutDefs = {'gtFile':'./data/Beirut/GroundTruth/allDamages.shp',
                 'crs':'EPSG:4326',
                 'labelColumn':'decision',
                 'map':True,
                 'mapOption':False,
                 'lat':33.893,
                 'lon':35.512,
                 'zoom':14,
                 'dataTypes':['High resolution imagery','Interferometry'],
                 'dataFiles':["./data/Beirut/HighResolution/20JUL31_HR_LatLon.tif","./data/Beirut/HighResolution/20AUG05_HR_LatLon.tif","./data/Beirut/InSAR/beirutPrePreExplosionIfg.tif","./data/Beirut/InSAR/beirutPrePostExplosionIfg.tif"],
                 'polyBounds':[35.512407312180784, 33.892243658425194, 35.52800151162857, 33.90124240848098],
                 'colors':['green','yellow','cyan','red','maroon']
                 }
    # Houston
    houstonDefs = {'gtFile':'./data/Houston/GroundTruth/2018_IEEE_GRSS_DFC_GT_TR.tif',
                  'crs':'EPSG:26915',
                  'labelColumn':'class',
                  'map':False,
                  'dataTypes':['Hyperspectral imagery','LiDAR','High resolution imagery'],
                  'dataFiles':[None, './data/Houston/Hyperspectral/20170218_UH_CASI_S4_NAD83.pix',None,'./data/Houston/LiDAR/UH17_GI3F051.tif', None,'./data/Houston/HighResolution/houstonmosaic.tif'],
                   'rmvClass':'0'
                  }
    # Other
    noneDefs = {'gtFile':None,
                'crs':'EPSG:4326',
                'labelColumn':'class',
                'map':True,
                'mapOption':True,
                'lat':0,
                'lon':0,
                'zoom':14,
                'dataTypes':[],
                'dataFiles':[]
                }
    
    box_layout = ipw.Layout(display='flex', width='900px', flex_flow='row')
    
    if bxDefaults.trait_values()['children'][1].value is options[0]: defs=beirutDefs
    elif bxDefaults.trait_values()['children'][1].value is options[1]: defs=houstonDefs
    else: defs=noneDefs
    
    ## Label Imports
    display(ipw.HTML(value = f"<b>{'Label parameters'}</b>"))
    # Label source
    bxGround = ipw.Box([ipw.Label(value='Ground truth: Shapefile - '), ipw.Text(value=defs['gtFile'], placeholder='groundtruth.shp',  disabled=False,layout=layout),
                       ipw.Label(value='Coordinate system - '), ipw.Text(value=defs['crs'], placeholder='EPSG:4326',  disabled=False,layout=layout)])
    bxLabel = ipw.Box([ipw.Label(value='Label column - '), ipw.Text(value=defs['labelColumn'], placeholder='label column',  disabled=False, layout=layout),
                       ipw.Label(value='First word only - '), ipw.Checkbox(value=False, disabled=False, indent=False)])
    
    # Confidence in label assignment
    bxConf = ipw.Box([ipw.Label(value='Label confidence ($P_{other label}$, $P_{class}$)'),
                      ipw.FloatRangeSlider(value=[0, 1], min=0, max=1, step=0.01, disabled=False, continuous_update=True, orientation='horizontal', readout=True, readout_format='.2f')])
    display(bxGround,bxLabel,bxConf)

    # Ask for map parameters input
    if defs['map']:
        display(ipw.HTML(value = f"<b>{'Map properties'}</b>"))
        # Define Map entries for display after confirmation
        bxMap = ipw.Box([ipw.Label(value='Latitude - '), ipw.FloatText(value=defs['lat'], placeholder='dd.dddd',  disabled=False,layout=layout),
                           ipw.Label(value='Longitude - '), ipw.FloatText(value=defs['lon'], placeholder='dd.dddd',  disabled=False,layout=layout),
                           ipw.Label(value='Zoom - '), ipw.IntText(value=defs['zoom'], placeholder='dd.dddd',  disabled=False, layout=layout)])
        if defs['mapOption']: bxMapOpt = ipw.Box([ipw.Label(value='Interactive map'), ipw.Checkbox(value=True, disabled=False, indent=False)])
        else: bxMapOpt = ipw.Box([ipw.Label(value='Standard test area - '), ipw.Checkbox(value=False, disabled=False, indent=False)])
        display(bxMap,bxMapOpt)
        v.update({'bxMap':bxMap, 'bxMapOpt':bxMapOpt})
    
    ## Data imports
    display(ipw.HTML(value = f"<b>{'Data Parameters'}</b>"))
    # Data Type Entry
    ndt = len(defs['dataTypes'])
    bxDataTypes = ipw.Box([ipw.Label(value='Enter Data Types:'),
                           ipw.Combobox(value=defs['dataTypes'][0] if ndt > 0 else None, placeholder='Data Type 1', options=defs['dataTypes'], ensure_option=False, disabled=False,layout=layout),
                           ipw.Combobox(value=defs['dataTypes'][1] if ndt > 1 else None, placeholder='Data Type 2', options=defs['dataTypes'], ensure_option=False, disabled=False,layout=layout),
                           ipw.Combobox(value=defs['dataTypes'][2] if ndt > 2 else None, placeholder='Data Type 3', options=defs['dataTypes'], ensure_option=False, disabled=False,layout=layout),
                           ipw.Combobox(value=defs['dataTypes'][3] if ndt > 3 else None, placeholder='Data Type 4', options=defs['dataTypes'], ensure_option=False, disabled=False,layout=layout)])
    
    # Button for confirmation and display next entries
    button1 = ipw.Button(description='Confirm Types', disabled=False, button_style='success', tooltip='Confirm', icon='check')
    
    display(bxDataTypes)

    # Once confirmation clicked display following
    out1 = ipw.Output()
    def on_button1_clicked(b1):
        button1.description = 'Confirmed'
        with out1:
            # Fetch confirmed data types
            print('--------------')
            dataTypes = [i.value.split(' ')[0] for i in bxDataTypes.trait_values()['children'][1:] if len(i.value) > 0]
            # Ask for input of file paths
            for i in range(len(dataTypes)):
                try: globals()['bxfile'+str(i)] = ipw.Box([ipw.Label(value=dataTypes[i]+' File Locations: Pre -'), ipw.Text(value=defs['dataFiles'][2*i], placeholder=dataTypes[i]+'PreFile', disabled=False),
                         ipw.Label(value=' Post -'), ipw.Text(value=defs['dataFiles'][2*i+1], placeholder=dataTypes[i]+'PostFile', disabled=False)])
                except: globals()['bxfile'+str(i)] = ipw.Box([ipw.Label(value=dataTypes[i]+' File Locations: Pre -'), ipw.Text(placeholder='Enter file path', disabled=False),
                         ipw.Label(value=' Post -'), ipw.Text(placeholder='Enter file path', disabled=False)])
                display(globals()['bxfile'+str(i)])
                v['bxfile'+str(i)] = globals()['bxfile'+str(i)]
    
    # Display confirmation button
    button1.on_click(on_button1_clicked)
    display(ipw.VBox([button1, out1]))
    
    # Update variables
    v.update({'bxGround':bxGround,'bxLabel':bxLabel,'bxConf':bxConf,'bxDataTypes':bxDataTypes,'defs':defs,'box_layout':box_layout})

    return v
  

#---------------------------------------------#
"""Model Parameter functions"""
# Display map with label data and boundary box for modelling area
def interactive_label_map(v, labels):
    # Extract parameter values from previous entries
    v['lat'], v['lon'], v['zoom'] = [i.value for i in v['bxMap'].trait_values()['children'][1::2]]
    if v['defs']['mapOption'] is False: v['stdTest'] = v['bxMapOpt'].trait_values()['children'][1].value
    for i in v.keys(): globals()[i] = v[i] # Extract variables for use

    # Display map of assessments upon which to draw Polygon for analysis
    m1 = pl.create_map(lat, lon, zoom)
    m1 = pl.plot_labels(labels, m1, cn=cn, classes=False, colors=defs['colors'] if 'colors' in defs.keys() else False, layer_name='Ground data')
    m1, testPoly = pl.draw_polygon(labels, m1, stdTest=stdTest if 'stdTest' in v.keys() else False, bounds=defs['polyBounds'] if 'polyBounds' in defs.keys() else False)
    display(m1)
    
    # Update variables
    v.update({'testPoly': testPoly, 'm1':m1})
    return v    

# Ask for entry of model parameters
def model_parameters(v):
    v['groundTruth'], v['crs'] = [i.value for i in v['bxGround'].trait_values()['children'][1::2]]
    v['cn'], v['splitString'] = [i.value for i in v['bxLabel'].trait_values()['children'][1::2]]
    v['dataTypes'] = [i.value.split(' ')[0] for i in v['bxDataTypes'].trait_values()['children'][1:] if len(i.value) > 0]
    for i in v.keys(): globals()[i] = v[i] 
        
    # Display map of labelling to define area of model
    # Import Labels to geodataframe from shapefile for interactive map
    if v['defs']['map'] is True and (v['defs']['mapOption'] is False or v['bxMapOpt'].trait_values()['children'][1].value is True):
        try: 
            labels = ip.shape_to_gdf(groundTruth, cn, splitString, crs=crs)
            v = interactive_label_map(v,labels)
        except:
            print('Interactive map not available with label format. Trying static map.')
            try: 
                labels, labelarr = ip.raster_to_df(groundTruth, cn=cn)
                pl.static_label_plot(labelarr)
            except: raise ValueError("Could not read labels into dataframe. Try different format (e.g. .shp, .tif)")
    else: 
        try: 
            labels, labelarr = ip.raster_to_df(groundTruth, cn=cn)
            pl.static_label_plot(labelarr)
        except: raise ValueError("Could not read labels into dataframe. Try different format (e.g. .shp, .tif)")
    for i in v.keys(): globals()[i] = v[i]    
    
    # Fetch variables from previous entries
    display(ipw.HTML(value = f"<h3>{'Model Parameters'}</h3>"))
    
    # Nodes
    bxNodes = ipw.Box([ipw.Label(value='Max nodes - '), ipw.IntText(value=10000, placeholder='10000',  disabled=False, step=1000, layout=layout, min=2),
                      ipw.Label(value='Ground Truth % - '), 
                      ipw.FloatSlider(value=50, min=0.1, max=99.9, step=0.1, disabled=False, orientation='horizontal', readout=True, readout_format='.1f')])

    # Edges
    # Neighbours for each data type
    bxEdges = ipw.Box([ipw.IntText(value=2, placeholder='edges', description=str(i)+' - ', disabled=False, layout=layout) for i in dataTypes],layout=box_layout)
    # Geographical neighbours
    bxAdjacent = ipw.Box([ipw.Label(value='Geographical Edges - '), ipw.Checkbox(value=False, disabled=False, indent=False, layout=layout), ipw.Label(value='Geographical Neighbours - '), ipw.IntText(value=2, placeholder='edges',  disabled=False, layout=layout)])
    
    # Nodes
    display(ipw.HTML(value = f"<b>{'Node Properties - Sampling occurs if max nodes < input nodes.'}</b>"))
    display(bxNodes)

    # Edges
    display(ipw.HTML(value = f"<b>{'Edge Properties - Number of neighbours each node is connected to according to input types.'}</b>"))
    display(bxEdges,bxAdjacent)
    
    display(ipw.HTML(value = f"<b>{'Class Properties'}</b>"))
    # Display default classes from labels
    unique = np.sort(labels[cn].unique())
    display(ipw.HTML(value = "Labels - "f"{str(unique)}"))
    # Ask if removing any classes
    bxRemove = ipw.Box([ipw.Label(value='Remove labels - '), ipw.Text(value=defs['rmvClass'] if 'rmvClass' in defs.keys() else '', placeholder='label1,label2',  disabled=False, layout=layout)])
    
    # Ask for number of classes to use
    bxNClasses = ipw.Box([ipw.Label(value='Classes for Model - '), ipw.Dropdown(options=list(range(2,len(unique)+1)),value=max(list(range(len(unique)+1))),disabled=False)])   
    display(bxRemove,bxNClasses)
    
    # Once confirmed then display classification options
    button3 = ipw.Button(description='Confirm Classes', disabled=False, button_style='success', tooltip='Confirm', icon='check')
    out3 = ipw.Output()
    def on_button3_clicked(b3):
        button3.description = 'Confirmed'
        with out3:
            # Read number of classes
            nClasses = bxNClasses.trait_values()['children'][1].value
            # If class grouping required propose options
            if nClasses < len(unique):
                print('-------------')
                # Opt to use clustering or not
                bxCluster = ipw.Box([ipw.Label(value='Use class clustering - Uncheck to assign classes below:'), ipw.Checkbox(value=True, disabled=False, indent=False)])
                # Assign each value to a class
                bxAssign = ipw.Box([ipw.SelectMultiple(options=unique, rows=len(unique), description='Class '+str(i)+':', disabled=False) for i in range(nClasses)], layout=box_layout)
                # Edit class names if desired
                bxClNames = ipw.Box([ipw.Text(value='cl'+str(i), placeholder='Enter Class Name', description='Class '+str(i)+':', disabled=False) for i in range(nClasses)],layout=box_layout)
                display(bxCluster, bxAssign)
                display(ipw.HTML(value = "Edit Class Names:"))
                display(bxClNames)
                
                v.update({'bxCluster':bxCluster, 'bxAssign':bxAssign, 'bxClNames':bxClNames})
                # PCA options if needed in future
                # pca, pcaComps, meanCluster = False, 2, True # Clustering properties if used
    
    # Display confirmation button
    button3.on_click(on_button3_clicked)
    display(ipw.VBox([button3, out3]))
    
    # Update variables
    v.update({'bxNClasses':bxNClasses, 'bxRemove':bxRemove, 'bxNodes':bxNodes,'bxEdges':bxEdges,'bxAdjacent':bxAdjacent,'unique':unique,'labels': labels})
    return v  
  
#------------------------------------#
"""Data import functions"""

# Reproject data if not in correct crs or single file
def reproject_data(v):
    print("------Checking Coordinate Systems-------")
    # Loop through each data type for files
    for i in range(len(v['dataTypes'])):
        # Check if it is a directory - try mosaicing components
        if os.path.isdir(v['postFile'+str(i)]): 
            print(v['postFile'+str(i)]+'is directory - Attempting to mosaic directory contents.')
            try: 
                v['postFile'+str(i)] = ip.get_training_array(ip.mosaic(v['groundTruth'],v['postFile'+str(i)]), v['groundTruth'])
                print('Mosaic successful')
            except: raise ValueError("Failed to mosaic directory")
        elif 'preFile'+str(i) in v.keys() and os.path.isdir(v['preFile'+str(i)]): v['preFile'+str(i)] = ip.get_training_array(get_mosaic(v['groundTruth'],v['preFile'+str(i)]), v['groundTruth']) 
                     
        # Check coordinate system is correct - otherwise reproject
        elif v['crs'] not in ip.get_crs(v['postFile'+str(i)]) and 'None' not in ip.get_crs(v['postFile'+str(i)]):
            v['postFile'+str(i)] = ip.conv_coords([v['postFile'+str(i)]], ["data/PostConv"+str(i)+".tif"], v['crs'])[0]
            if v['preFile'+str(i)]: v['preFile'+str(i)] = conv_coords(v['preFile'+str(i)], ["data/PreConv"+str(i)+".tif"], v['crs'])[0]
    print("------Finished Checking Coordinate Systems-------")
    return v

# Import datatypes into dataframe
def import_data(v):
    # Retrieve file locations from inputs
    v['dataTypes'] = [i.value.split(' ')[0] for i in v['bxDataTypes'].trait_values()['children'][1:] if len(i.value) > 0]
    for j in range(len(v['dataTypes'])):
        try: v['preFile'+str(j)], v['postFile'+str(j)] = [i.value for i in v['bxfile'+str(j)].trait_values()['children'][1::2]]    
        except KeyError: raise KeyError('Please make sure you have confirmed the data types.')
   
    # Reproject Data if necessary
    v = reproject_data(v)
    for i in v.keys(): globals()[i] = v[i] # Retrieve variables to use
        
    # Import Files
    print("------Importing Data Files---------")
    # Import first data type
    if defs['map']:
        globals()['dataArray0'], crop = ip.img_to_df(postFile0, testPoly, crs=crs)
        if preFile0:
            preDf, _ = ip.img_to_df(preFile0, testPoly, crs=crs)
            globals()['dataArray0'] -= preDf

        # Import other data types
        if len(dataTypes) > 1:
            crop.rio.to_raster("croptemp.tif")
            for i in range(1, len(dataTypes)):
                ip.resample_tif(globals()['postFile'+str(i)], testPoly, 'posttemp'+str(i)+'.tif')
                globals()['dataArray'+str(i)] = ip.tif_to_df('posttemp'+str(i)+'.tif', 'resample')
                if globals()['preFile'+str(i)]: 
                    ip.resample_tif(globals()['preFile'+str(i)], testPoly, 'pretemp'+str(i)+'.tif')
                    globals()['dataArray'+str(i)] -= ip.tif_to_df('pretemp'+str(i)+'.tif', 'resample')
            ip.del_file_endings(".", "temp*.tif")
            
    # If first file not needed for cropping raster
    else: 
        # Loop through datatypes
        for i in range(len(dataTypes)):
            # If postfile is file
            if type(globals()['postFile'+str(i)]) is str and os.path.isfile(globals()['postFile'+str(i)]):
                # Last minute Houston addition if - to avoid mosaicing for demo
                if 'houstonmosaic' in globals()['postFile'+str(i)]: 
                    globals()['dataArray'+str(i)] = ip.arr_to_df(np.array(imageio.imread(globals()['postFile'+str(i)])).reshape(3,1202,4768))
                else: globals()['dataArray'+str(i)], _ = ip.raster_to_df(globals()['postFile'+str(i)], cn=dataTypes[i], crop=True, target=groundTruth)
                print('Imported '+globals()['postFile'+str(i)])
            # Else post file is a mosaiced directory now an array
            else: globals()['dataArray'+str(i)] = ip.arr_to_df(globals()['postFile'+str(i)])
            # Check for pre files and do the same
            if globals()['preFile'+str(i)]: 
                if type(globals()['preFile'+str(i)]) is str and os.path.isfile(globals()['preFile'+str(i)]):
                    df_temp, _ = ip.raster_to_df(globals()['preFile'+str(i)], cn=dataTypes[i], crop=True, target=groundTruth)
                    globals()['dataArray'+str(i)] -= df_temp
                    print('Imported '+globals()['preFile'+str(i)])
                else: globals()['dataArray'+str(i)] -= ip.arr_to_df(globals()['preFile'+str(i)])
    
    # Get the column names separated into sublists for each type, to be used later
    typesUsed=[]
    for j, val in enumerate(dataTypes):
        globals()['dataArray'+str(j)].columns = [str(val)+s if str(val) not in s else s for s in list(globals()['dataArray'+str(j)].columns.values.astype(str))]
        globals()['dataArray'+str(j)].index = globals()['dataArray'+str(0)].index # All data is with dataArray0 coords (choose wisely)
        typesUsed += [list(globals()['dataArray'+str(j)].columns.values)]
    
    # Concatenate dataframes
    data = ip.concat_dfs([globals()['dataArray'+str(i)] for i in range(len(dataTypes))])
  
    print("------Finished Importing Files---------")

    v.update({'data':data, 'typesUsed':typesUsed})
    return v
  
#------------------------------------------#
"""Data classification functions"""
  
def classify_data(v,seed=1):
    # Retrieve data from inputs
    for i in v.keys(): globals()[i] = v[i]
    max_nodes = bxNodes.trait_values()['children'][1].value
    nClasses = bxNClasses.trait_values()['children'][1].value
    classAssign = False if ('bxAssign' not in v) or (bxCluster.trait_values()['children'][1].value is True) else [list(i.value) for i in bxAssign.trait_values()['children']]
    classNames = False if 'bxClNames' not in v else [i.value for i in bxClNames.trait_values()['children']]
    rmvClass = bxRemove.trait_values()['children'][1].value.split(',')

    # Sample data and create geodataframe
    print("------Data Sampling---------")
    if max_nodes < 2: raise ValueError("Insufficient Nodes for belief propagation")
    gdf = tr.get_sample_gdf(data, max_nodes, crs,seed=1)
    
    # Sample labels
    if 'GeoDataFrame' not in str(type(labels)):
        labelsUsed = tr.get_sample_gdf(labels, max_nodes, crs,seed=1)
    else: labelsUsed = labels.copy().to_crs(crs)
    if rmvClass: 
        kept = [i not in rmvClass for i in labelsUsed[cn].astype(str)]
        labelsUsed = labelsUsed.iloc[kept].reset_index() # Remove undesire labels (e.g. unclassified data)
        if 'GeoDataFrame' not in str(type(labels)):
            gdf = gdf.iloc[kept].reset_index()
    print("------Data Classification---------")
    
    defClasses, dataUsed = len(labelsUsed[cn].unique()), gdf.copy() # Default classes from labels
    usedNames = labelsUsed[cn].unique() if nClasses==defClasses or nClasses is False else classNames
    initial = tr.init_beliefs(dataUsed, classes=nClasses, columns=usedNames, crs=crs) # Initial class value for each data pixel

    if not nClasses or nClasses == defClasses: 
        nClasses = defClasses # If default classes used
        classesUsed = usedNames.copy()
    elif nClasses > defClasses: 
        raise NameError('Cannot assign more classes than in original data') # If invalid input
    elif nClasses < defClasses: # Perform class grouping
        items = [item for sublist in classAssign for item in sublist] if classAssign is not False else False
        if (classAssign is False) or not any(classAssign) or (len(items) is not (len(set(items)))): # Perform clustering
            if classAssign is not False: print('Incorrect class assignment - Proceeding with clustering. Please assign a single class for each value.')
            # Assign labels to each pixel
            if 'GeoDataFrame' in str(type(labels)): 
                allPixels = tr.create_nodes(initial, labelsUsed[['geometry',cn]][labelsUsed.within(tr.get_polygon(testPoly, conv=True))])
            else: allPixels = tr.create_nodes(initial, labelsUsed[['geometry',cn]])
            # Run PCA if set to True
            #X = hf.run_PCA(dataUsed[typesUsed[0]].values.transpose(), pcaComps).components_.transpose() if pca else dataUsed[typesUsed[0]]
            types = [item for sublist in typesUsed for item in sublist]
            X = dataUsed[types]
            #print(allPixels[cn].dropna().index)
            # Run clustering
            meanCluster = True
            kmeans, clusterClasses, initLabels = tr.run_cluster(X.loc[allPixels[cn].dropna().index].values.reshape(-1,len(types)), allPixels[cn].dropna(), meanCluster, nClasses)
            print('Clustered classes:{} , original classes:{}'.format(clusterClasses, initLabels))
            # Create groups of classes
            classesUsed = []
            for j in range(nClasses): classesUsed.append([initLabels[i] for i, x in enumerate(list(clusterClasses)) if x==j])
        
        else:
            if len(set(items)) is not defClasses:
                print('Not all labels have been assigned to class. Sampling data to include only labels selected.')
                labelsUsed = labelsUsed.loc[labelsUsed[cn].isin(items)]
            classesUsed = classAssign
            #used = [i in flatten_list(classesUsed) for i in labelsUsed[cn]]
            initial = tr.init_beliefs(dataUsed, classes=nClasses, columns=usedNames, crs=crs)

        # Assign labels for each pixel after clustering
        labelsUsed[cn] = tr.group_classes(labelsUsed[cn], classesUsed)
    print("------Finished Data Classification---------") 

    bxBalance = ipw.Box([ipw.Label(value='Balance classes - '), ipw.Checkbox(value=True if nClasses==2 else False, disabled=False, indent=False)])
    bxLimit = ipw.Box([ipw.Label(value='Loss function limit (logarithmic) - '), ipw.FloatLogSlider(value=1e-3, base=10, min=-15, max=0, step=1)])
    display(bxBalance, bxLimit)
    
    # Update variables
    v.update({'max_nodes':max_nodes, 'nClasses':nClasses, 'classAssign':classAssign,'classNames':classNames, 'labelsUsed':labelsUsed,'initial':initial, 'usedNames':usedNames, 'classesUsed':classesUsed, 'dataUsed':dataUsed, 'bxBalance':bxBalance, 'bxLimit':bxLimit})
    return v
  
#------------------------------------------#
"""Run Belief Propagation"""
def run_bp(v):
    # Retrieve data from inputs
    for i in v.keys(): globals()[i] = v[i]
    initial = v['initial'].copy()
    trainSplit = bxNodes.trait_values()['children'][3].value
    confidence = list(bxConf.trait_values()['children'][1].value)
    neighbours = [i.value for i in bxEdges.trait_values()['children']]
    adjacent, geoNeighbours = [i.value for i in bxAdjacent.trait_values()['children'][1::2]]
    equivUse = bxBalance.trait_values()['children'][1].value
    limit = bxLimit.trait_values()['children'][1].value

    # Split pixels in to train and test sets    
    X_train, X_test, y_train, y_test = tr.train_test_split(labelsUsed, cn, tr.get_polygon(testPoly, conv=True) if 'testPoly' in v.keys() else False, testSplit=(1-(trainSplit/100)))
    
    # Create nodes
    nodes = tr.create_nodes(initial, X_train)

    summary = nodes.groupby(cn).size()
    if equivUse:
        equiv = gpd.GeoDataFrame()
        for i in summary.index.values:
            equiv = equiv.append(nodes[nodes[cn] == i][0:min(summary)])
        equiv = equiv.append(nodes[[np.isnan(x) for x in nodes[cn]]])
        nodes=equiv.copy()
        initial = initial.loc[nodes.index.values].reset_index()
    
    # Assign prior beliefs from assessments
    priors = tr.prior_beliefs(nodes, beliefColumns = initial.columns[-nClasses:], beliefs=confidence, classNames=classNames, column = cn)
    
    if all(values is 0 for values in neighbours) and (geoNeighbours is 0):
        edges, beliefs = [], priors
    else:
        # Create edges
        edges = tr.create_edges(nodes, adjacent=adjacent, geo_neighbours=geoNeighbours, values=typesUsed, neighbours=neighbours)

        # Run belief propagation
        beliefs, _ = nc.netconf(edges,priors,verbose=True,limit=limit)
    
    v.update({'trainSplit':trainSplit, 'confidence':confidence, 'neighbours':neighbours, 'adjacent':adjacent, 'geoNeighbours':geoNeighbours, 'X_train':X_train, 'X_test':X_test, 'nodes':nodes, 'priors':priors, 'edges':edges,'beliefs':beliefs,'initial':initial, 'equivUse':equivUse, 'limit':limit})
    return v
  

#------------------------------------------------#
"""Evaluation Metrics"""
def evaluate_output(v):
    for i in v.keys(): globals()[i] = v[i]
    equivTest = equivUse
    # Get y_true vs y_pred for test set
    y_true, y_pred = tr.get_labels(initial, X_test, beliefs, column=cn, equivTest=equivTest)
    
    # Classification metrics
    true_clf, pred_clf = tr.class_metrics(y_true, y_pred, classes=usedNames, orig=unique)

    fig, axs = pl.create_subplots(1,2, figsize=[12,5])
    
    # Confusion matrix
    axs = pl.confusion_matrix(axs, true_clf, pred_clf, usedNames)

    # Cross entropy / Confidence metrics
    if nClasses == 2: axs = pl.cross_entropy_metrics(axs, y_true, y_pred[:,1].reshape(-1,1), usedNames)
    
    else: axs[1] = pl.cross_entropy_multiclass(axs[1], true_clf, y_pred, usedNames)

    pl.show_plot()
    
    v.update({'y_true':y_true, 'y_pred':y_pred, 'true_clf':true_clf, 'pred_clf':pred_clf, 'fig':fig, 'equivTest':equivTest})
    
    return v
  
# Save figure
def save_plot(v, location=False):
    for i in v.keys(): globals()[i] = v[i]
    if location: pl.save_plot(fig, location)
    else: pl.save_plot(fig, 'results/Beirut_UN_nd{}tr{}_cls{}{}_neighbours{}{}_std{}_adj{}{}'.format(str(len(nodes)), str(int(trainSplit*100)), str(nClasses),str(classesUsed),
                                                                                          str(dataTypes),str(neighbours),str(stdTest),
                                                                                          str(adjacent),str(geoNeighbours)))
      
      
#----------------------------------------------------#
import ipyleaflet as ipl
"""Result map plotting"""
# This bit is not quite as well functioned off.
# Visualise spatial results
def map_result(v):
    for i in v.keys(): globals()[i] = v[i] # Retrieve variables to use
    if nClasses > 2: 
        print('Not yet supported for more than two classes.')
        return
    ngrid=100 # Gridding for contour maps
    
    # Sample for test locations
    tests = gpd.sjoin(initial, X_test, how='left', op='within').dropna(subset=[cn])
    summary = tests.groupby(cn).size()
    if equivTest:
        equiv = gpd.GeoDataFrame()
        for i in summary.index.values:
            equiv = equiv.append(tests[tests[cn] == i][0:min(summary)])
        equiv = equiv.append(tests[[np.isnan(x) for x in tests[cn]]])
        tests = equiv.copy()
    tests['prediction']=pred_clf

    # Plot interactive map for shape type labels
    if 'GeoDataFrame' in str(type(labels)):
        # Create map
        mf = pl.create_map(lat, lon, zoom, basemap=ipl.basemaps.OpenStreetMap.BlackAndWhite)

        # Plot ground truth
        pl.plot_assessments(labels, mf, cn=cn, layer_name='Ground truth', fill=0.3, legName='Ground Truth')

        # Plot training locations
        pl.plot_assessments(nodes.to_crs({'init':crs}).dropna(), mf, layer_name='Train Locations', no_leg=True, classes=sorted([x for x in nodes.decision.unique() if str(x) != 'nan']), colors = ['green', 'red'] if nClasses==2 else None)

        # Plot test locations
        pl.plot_assessments(tests.to_crs({'init':crs}).dropna(), mf, cn='prediction', layer_name='Test Predictions', no_leg=True, classes=[x for x in tests.prediction.unique() if str(x) != 'nan'], colors = ['green', 'red'] if nClasses==2 else None)

        # Create contours for predictions 
        xi, yi = np.linspace(nodes.geometry.x.min(), nodes.geometry.x.max(), ngrid), np.linspace(nodes.geometry.y.min(), nodes.geometry.y.max(), ngrid)
        zi = griddata((nodes.geometry.x, nodes.geometry.y), (beliefs[:,0]-beliefs[:,1]+0.5), (xi[None, :], yi[:, None]), method='nearest')
        cs = plt.contourf(xi, yi, zi, levels=math.floor((zi.max()-zi.min())/0.1)-1, extend='both')
        plt.close() 

        # Colours are added for each layer, unfortunately quite manual just now
        colorsRed = ['#e50000','#ff0000','#ff3232','#ff6666','#ff9999']
        colorsGreen = ['#e5ffe5','#b2f0b2','#99eb99','#66e166','#32d732','#00b800'] if (v['bxNodes'].trait_values()['children'][3].value < 15) else ['#b2f0b2','#99eb99','#66e166','#32d732','#00b800']
        colors=[]
        for i in range(math.floor(len(cs.allsegs)/2-6)-math.floor(((zi.max()-1-(0-zi.min()))/0.1)/2)): colors.append('#ff0000')
        colors += colorsRed
        colors += colorsGreen
        for i in range(math.ceil(len(cs.allsegs)/2-4)+math.floor(((zi.max()-1-(0-zi.min()))/0.1)/2)): colors.append('#32d732')

        # Add each contour layer as polygon map layer
        allsegs, allkinds = cs.allsegs, cs.allkinds
        contourLayer = ipl.LayerGroup(name = 'Assessment Contours')
        for clev in range(len(cs.allsegs)):
            kinds = None if allkinds is None else allkinds[clev]
            segs = pl.split_contours(allsegs[clev], kinds)
            polygons = ipl.Polygon(locations=[p.tolist() for p in segs], color=colors[clev],
                                   weight=1, opacity=0.5, fill_color=colors[clev], fill_opacity=0.4,
                                   name='layer_name')
            contourLayer.add_layer(polygons)
        mf.add_layer(contourLayer)

        # Add layer control
        control = ipl.LayersControl(position='topright')
        mf.add_control(control)

        # Add colors legend
        leg = dict(zip([str(round(x-0.1,1))+'-'+str(round(x,1)) for x in np.linspace(1,0.1,10).tolist()],colorsRed+colorsGreen))
        l2 = ipl.LegendControl(leg, name='Damage Prob', position="topleft")
        mf.add_control(l2)

        # Add widgets to map
        zoom_slider = ipw.IntSlider(description='Zoom level:', min=7, max=18, value=14)
        ipw.jslink((zoom_slider, 'value'), (mf, 'zoom'))
        widget_control1 = ipl.WidgetControl(widget=zoom_slider, position='topright')
        mf.add_control(widget_control1)
        mf.add_control(ipl.FullScreenControl(position='topright'))
        mf.zoom_control = False

        # Display map
        display(mf)
        return mf

    # Plot static map for image style labelling
    else:
        fig, [ax1,ax2] = pl.create_subplots(2,1,[10,5])
        normalized = (beliefs[:,1]-min(beliefs[:,1]))/(max(beliefs[:,1])-min(beliefs[:,1]))
        res = ax1.tricontourf(nodes.geometry.x,nodes.geometry.y,normalized,cmap='RdYlGn',levels=10)
        cb1 = fig.colorbar(res,ax=ax1)
        cb1.set_label('Class 1 probability',fontsize=12), ax1.set_title('Class Probability Map',size=14)
        ax1.invert_yaxis()
        ax1.set_xlabel('x pixels', fontsize=12), ax1.set_ylabel('y pixels',fontsize=12)
        ax2.set_xlabel('x pixels', fontsize=12), ax2.set_ylabel('y pixels',fontsize=12)

        ax2.invert_yaxis()

        a = ax2.tricontourf(labelsUsed.dropna().geometry.x,labelsUsed.dropna().geometry.y, labelsUsed['class'].values,alpha=0.7,levels=1, cmap='RdYlGn')
        a2 = ax2.tricontourf(labelsUsed.dropna().geometry.x,labelsUsed.dropna().geometry.y, labelsUsed['class'].values,alpha=0.7,levels=2, cmap='RdYlGn')

        ds = 5 # Downsample
        ax2.scatter(tests[::ds].geometry.x,tests[::ds].geometry.y,c='r',label=classNames[0])
        ax2.scatter(tests[::ds].geometry.x,tests[::ds].geometry.y,c='g',label=classNames[1])
        ax2.scatter(tests[::ds].geometry.x,tests[::ds].geometry.y,c=[classNames.index(i) for i in pred_clf][::ds],cmap='RdYlGn')
        cb2 = fig.colorbar(a, ax=ax2, ticks=[0,1])
        cb2.set_label('Ground Truth Class',fontsize=12), ax2.set_title('Test Prediction Map',size=14)
        ax2.legend(title='Predictions',loc='lower left')
        fig.tight_layout()
        return(fig)

