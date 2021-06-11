import ipywidgets as ipw
# Import helper functions
import imports as ip
import netconf as nc
import plotting as pl
import helper_functions as hf

## Label and data parameter setting
def parameter_input():
    ## Label Imports
    layout = {'width': 'max-content'}
    display(ipw.HTML(value = f"<b>{'Label Parameters'}</b>"))
    # Label source
    bxLabel = ipw.Box([ipw.Label(value='Damage Labels: Shapefile - '), ipw.Text(value='./data/beirutDamages.shp', placeholder='damages.shp',  disabled=False,layout=layout),
                       ipw.Label(value='Coordinate System - '), ipw.Text(value='EPSG:4326', placeholder='EPSG:4326',  disabled=False,layout=layout),
                       ipw.Label(value='Decision Column - '), ipw.Text(value='decision', placeholder='decision',  disabled=False, layout=layout),
                       ipw.Label(value='First Word Only - '), ipw.Checkbox(value=False, disabled=False, indent=False)])
    # Confidence in label assignment
    bxConf = ipw.Box([ipw.Label(value='Label Confidence ($P_{other label}$, $P_{class}$)'),
                      ipw.FloatRangeSlider(value=[0, 1], min=0, max=1, step=0.01, disabled=False, continuous_update=True, orientation='horizontal', readout=True, readout_format='.2f')])
    display(bxLabel,bxConf)

    # Ask for map parameters input
    display(ipw.HTML(value = f"<b>{'Map Properties'}</b>"))
    # Define Map entries for display after confirmation
    bxMap = ipw.Box([ipw.Label(value='Latitude - '), ipw.FloatText(value=33.893, placeholder='dd.dddd',  disabled=False,layout=layout),
                       ipw.Label(value='Longitude - '), ipw.FloatText(value=35.512, placeholder='dd.dddd',  disabled=False,layout=layout),
                       ipw.Label(value='Zoom - '), ipw.IntText(value=14, placeholder='dd.dddd',  disabled=False, layout=layout),
                       ipw.Label(value='Standard Test Area - '), ipw.Checkbox(value=False, disabled=False, indent=False)])
    display(bxMap)
    
    ## Data imports
    display(ipw.HTML(value = f"<b>{'Data Parameters'}</b>"))
    # Data Type Entry
    bxDataTypes = ipw.Box([ipw.Label(value='Enter Data Types:'),
                           ipw.Combobox(value='HighRes Imagery', placeholder='Data Type 1', options=['HighRes Imagery','Interferogram','LowRes Imagery'], ensure_option=False, disabled=False,layout=layout),
                           ipw.Combobox(value='Interferogram', placeholder='Data Type 2', options=['HighRes Imagery','Interferogram','LowRes Imagery'], ensure_option=False, disabled=False,layout=layout),
                           ipw.Combobox(placeholder='Data Type 3', options=['HighRes Imagery','Interferogram','LowRes Imagery'], ensure_option=False, disabled=False,layout=layout),
                           ipw.Combobox(placeholder='Data Type 4', options=['HighRes Imagery','Interferogram','LowRes Imagery'], ensure_option=False, disabled=False,layout=layout)])
    
    # Button for confirmation and display next entries
    button1 = ipw.Button(description='Confirm Types', disabled=False, button_style='success', tooltip='Confirm', icon='check')
    
    display(bxDataTypes)
    
    # Default data files
    defFiles = ["data/highRes/20JUL31_HR_LatLon.tif","data/highRes/20AUG05_HR_LatLon.tif","./data/beirutPrePreExplosionIfg.tif","./data/beirutPrePostExplosionIfg.tif"]
    
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
                try: globals()['bxfile'+str(i)] = ipw.Box([ipw.Label(value=dataTypes[i]+' File Locations: Pre -'), ipw.Text(value=defFiles[2*i], placeholder=dataTypes[i]+'PreFile', disabled=False),
                         ipw.Label(value=' Post -'), ipw.Text(value=defFiles[2*i+1], placeholder=dataTypes[i]+'PostFile', disabled=False)])
                except: globals()['bxfile'+str(i)] = ipw.Box([ipw.Label(value=dataTypes[i]+' File Locations: Pre -'), ipw.Text(placeholder='Enter file path', disabled=False),
                         ipw.Label(value=' Post -'), ipw.Text(placeholder='Enter file path', disabled=False)])
                display(globals()['bxfile'+str(i)])
                v['bxfile'+str(i)] = globals()['bxfile'+str(i)]
    
    # Display confirmation button
    button1.on_click(on_button1_clicked)
    display(ipw.VBox([button1, out1]))
    
    # Update variables
    v = {}
    v.update({'bxLabel':bxLabel,'bxConf':bxConf,'bxDataTypes':bxDataTypes,'bxMap':bxMap})

    return v
  
  
# Display map with label data and boundary box for modelling area
def label_map(v):
    # Extract parameter values from previous entries
    v['groundTruth'], v['crs'], v['cn'], v['splitString'] =  [i.value for i in v['bxLabel'].trait_values()['children'][1::2]]
    v['lat'], v['lon'], v['zoom'], v['stdTest'] = [i.value for i in v['bxMap'].trait_values()['children'][1::2]]
    for i in v.keys(): globals()[i] = v[i] # Extract variables for use

    # Import Labels to geodataframe for plotting
    labels = ip.shape_to_gdf(groundTruth, splitString, cn, crs=crs)

    # Display map of assessments upon which to draw Polygon for analysis
    m1 = pl.create_map(lat, lon, zoom)
    m1 = pl.plot_assessments(labels, m1, cn=cn)
    m1, testPoly = pl.draw_polygon(labels, m1, stdTest)
    display(m1)
    
    # Update variables
    v.update({'labels': labels, 'testPoly': testPoly})
    return v

# Ask for entry of model parameters
def model_parameters(v):  
    # Display map of labelling to define area of model
    v = label_map(v)
    
    # Fetch variables from previous entries
    v['dataTypes'] = [i.value.split(' ')[0] for i in v['bxDataTypes'].trait_values()['children'][1:] if len(i.value) > 0]
    for i in v.keys(): globals()[i] = v[i] # Extract variables for use
    
    layout = {'width': 'max-content'}
    display(ipw.HTML(value = f"<h3>{'Model Parameters'}</h3>"))
    
    # Nodes
    bxNodes = ipw.Box([ipw.Label(value='Maximum nodes - Sampling occurs if < pixel number: '), ipw.IntText(value=20000, placeholder='20000',  disabled=False, step=1000, layout=layout, min=2)])

    # Edges
    # Neighbours for each data type
    bxEdges = ipw.Box([ipw.Label(value='Neighbours - Edges to nearest values for each node: '), ipw.Box([ipw.IntText(value=3, placeholder='edges', description=str(i)+' - ', disabled=False, layout=layout) for i in dataTypes])])
    # Geographical neighbours
    bxAdjacent = ipw.Box([ipw.Label(value='Geographical Edges - '), ipw.Checkbox(value=False, disabled=False, indent=False, layout=layout), ipw.Label(value='Geographical Neighbours - '), ipw.IntText(value=4, placeholder='edges',  disabled=False, layout=layout)])
    
    # Nodes
    display(ipw.HTML(value = f"<b>{'Node Properties'}</b>"))
    display(bxNodes)

    # Edges
    display(ipw.HTML(value = f"<b>{'Edge Properties'}</b>"))
    display(bxEdges,bxAdjacent)
    
    display(ipw.HTML(value = f"<b>{'Class Properties'}</b>"))
    # Display default classes from labels
    unique = labels[cn].unique()
    display(ipw.HTML(value = "Label Classes - "f"{str(unique)}"))
    # Ask for number of classes to use
    bxNClasses = ipw.Box([ipw.Label(value='Classes for Model - '), ipw.Dropdown(options=list(range(2,len(unique)+1)),value=max(list(range(len(unique)+1))),disabled=False)])   
    display(bxNClasses)
    
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
                bxAssign = ipw.Box([ipw.SelectMultiple(options=unique, rows=len(unique), description='Class '+str(i)+':', disabled=False) for i in range(nClasses)])
                # Edit class names if desired
                bxClNames = ipw.Box([ipw.Text(value='cl'+str(i), placeholder='Enter Class Name', description='Class '+str(i)+':', disabled=False) for i in range(nClasses)])
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
    v.update({'bxNClasses':bxNClasses,'bxNodes':bxNodes,'bxEdges':bxEdges,'bxAdjacent':bxAdjacent,'unique':unique})
    return v  
  
  
# Data Imports
# Reproject data to used crs
def reproject_data(v):
    print("------Checking Coordinate Systems-------")
    for i in range(len(v['dataTypes'])):
        if v['crs'] not in ip.get_crs(v['postFile'+str(i)]):
            v['postFile'+str(i)] = ip.conv_coords([v['postFile'+str(i)]], ["data/PostConv"+str(i)+".tif"], v['crs'])[0]
            if v['preFile'+str(i)]: v['preFile'+str(i)] = ip.conv_coords(v['preFile'+str(i)], ["data/PreConv"+str(i)+".tif"], v['crs'])[0]
    print("------Finished Checking Coordinate Systems-------")
    return v

def import_data(v):
    # Retrieve file locations from inputs
    for j in range(len(v['dataTypes'])):
        try: v['preFile'+str(j)], v['postFile'+str(j)] = [i.value for i in v['bxfile'+str(j)].trait_values()['children'][1::2]]    
        except KeyError: raise KeyError('Please make sure you have confirmed the data types.')
    for i in v.keys(): globals()[i] = v[i] # Retrieve variables to use
   
    # Reproject Data if necessary
    v = reproject_data(v)
    
    # Import Files
    print("------Importing Data Files---------")
    # Import first data type
    df, crop = ip.img_to_df(postFile0, testPoly, crs=crs)
    if preFile0:
        preDf, _ = ip.img_to_df(preFile0, testPoly, crs=crs)
        df -= preDf

    # Import other data types
    if len(dataTypes) > 1:
        crop.rio.to_raster("croptemp.tif")
        for i in range(1, len(dataTypes)):
            ip.resample_tif(globals()['postFile'+str(i)], testPoly, 'posttemp'+str(i)+'.tif')
            globals()['dataArray'+str(i)] = ip.tif_to_array('posttemp'+str(i)+'.tif', 'resample')
            if globals()['preFile'+str(i)]: 
                ip.resample_tif(globals()['preFile'+str(i)], testPoly, 'pretemp'+str(i)+'.tif')
                globals()['dataArray'+str(i)] -= ip.tif_to_array('pretemp'+str(i)+'.tif', 'resample')
        ip.del_file_endings(".", "temp*.tif")

    # Concatenate data types
    data = df.copy()
    for j in range(1, len(dataTypes)): data[dataTypes[j]]=globals()['dataArray'+str(j)].flatten()
    data.dropna(inplace=True)
    print("------Finished Data Import---------")
    
    v.update({'data':data, 'typesUsed':[list(df.columns.values), dataTypes[1:]]})
    return v
  
## Assign Label classes to data
def classify_data(v):
    # Retrieve data from inputs
    for i in v.keys(): globals()[i] = v[i]
    max_nodes = bxNodes.trait_values()['children'][1].value
    nClasses = bxNClasses.trait_values()['children'][1].value
    classAssign = False if ('bxAssign' not in v) or (bxCluster.trait_values()['children'][1].value is True) else [list(i.value) for i in bxAssign.trait_values()['children']]
    classNames = False if 'bxClNames' not in v else [i.value for i in bxClNames.trait_values()['children']]

    # Sample data and create geodataframe
    print("------Data Sampling---------")
    if max_nodes < 2: raise ValueError("Insufficient Nodes for belief propagation")
    gdf = ip.get_sample_gdf(data, max_nodes, crs)
   
    print("------Data Classification---------")
    
    defClasses, labelsUsed, dataUsed = len(labels[cn].unique()), labels.to_crs(crs).copy(), gdf.copy() # Default classes from labels
    usedNames = labels[cn].unique() if nClasses==defClasses or nClasses is False else classNames
    initial = hf.init_beliefs(dataUsed, classes=nClasses, columns=usedNames, crs=crs) # Initial class value for each data pixel

    if not nClasses or nClasses == defClasses: 
        nClasses = defClasses # If default classes used
        classesUsed = usedNames.copy()
    elif nClasses > defClasses: raise NameError('Cannot assign more classes than in original data') # If invalid input
    elif nClasses < defClasses: # Perform class grouping
        if (classAssign is False) or not any(classAssign) or (len(set([item for sublist in classAssign for item in sublist])) is not defClasses) or (len([item for sublist in classAssign for item in sublist]) is not defClasses): # Perform clustering
            if classAssign is not False: print('Incorrect class assignment - Proceeding with clustering. Please assign a single class for each value.')
            # Assign labels to each pixel
            allPixels = hf.create_nodes(initial, labelsUsed[['geometry',cn]][labelsUsed.within(hf.get_polygon(testPoly, conv=True))])
            # Run PCA if set to True
            #X = hf.run_PCA(dataUsed[typesUsed[0]].values.transpose(), pcaComps).components_.transpose() if pca else dataUsed[typesUsed[0]]
            types = [item for sublist in typesUsed for item in sublist]
            X = dataUsed[types]
            # Run clustering
            meanCluster = True
            kmeans, clusterClasses, initLabels = hf.run_cluster(X.iloc[allPixels[cn].dropna().index].values.reshape(-1,len(types)), allPixels[cn].dropna(), meanCluster, nClasses)
            print('Clustered classes:{} , original classes:{}'.format(clusterClasses, initLabels))
            # Create groups of classes
            classesUsed = []
            for j in range(nClasses): classesUsed.append([initLabels[i] for i, x in enumerate(list(clusterClasses)) if x==j])
        else: 
            classesUsed = classAssign
            #used = [i in flatten_list(classesUsed) for i in labelsUsed[cn]]
            initial = hf.init_beliefs(dataUsed, classes=nClasses, columns=usedNames, crs=crs)

        # Assign labels for each pixel after clustering
        labelsUsed[cn] = hf.group_classes(labelsUsed[cn], classesUsed)
    print("------Finished Data Classification---------") 

    # Update variables
    v.update({'max_nodes':max_nodes, 'nClasses':nClasses, 'classAssign':classAssign,'classNames':classNames, 'labelsUsed':labelsUsed,'initial':initial, 'usedNames':usedNames, 'classesUsed':classesUsed})
    return v
  
def run_bp(v):
    # Retrieve data from inputs
    for i in v.keys(): globals()[i] = v[i]
    confidence = list(bxConf.trait_values()['children'][1].value)
    neighbours = [i.value for i in bxEdges.trait_values()['children'][1].trait_values()['children']]
    adjacent, geoNeighbours = [i.value for i in bxAdjacent.trait_values()['children'][1::2]]
    
    # Split train/test set for located nodes
    X_train, X_test, y_train, y_test = hf.train_test_split(labelsUsed, cn, hf.get_polygon(testPoly, conv=True))

    # Create nodes
    nodes = hf.create_nodes(initial, X_train)

    # Assign prior beliefs from assessments
    priors = hf.prior_beliefs(nodes, beliefColumns = initial.columns[-nClasses:], beliefs=confidence, classNames=classNames, column = cn)

    # Create edges
    edges = hf.create_edges(nodes, adjacent=adjacent, geo_neighbors=geoNeighbours, values=typesUsed, neighbours=neighbours)
    
    # Run belief propagation
    beliefs, _ = nc.netconf(edges,priors,verbose=True,limit=1e-3)
    
    v.update({'confidence':confidence, 'neighbours':neighbours, 'adjacent':adjacent, 'geoNeighbours':geoNeighbours, 'X_train':X_train, 'X_test':X_test, 'nodes':nodes, 'priors':priors, 'edges':edges,'beliefs':beliefs})
    return v
  
# Evaluation Metrics
def evaluate_output(v):
    for i in v.keys(): globals()[i] = v[i]
    # Get y_true vs y_pred for test set
    y_true, y_pred = hf.get_labels(initial, X_test, beliefs, column=cn)
    
    # Classification metrics
    pred_clf, true_clf = hf.class_metrics(y_true, y_pred, classes=usedNames, orig=unique)

    fig, axs = pl.create_subplots(1,2, figsize=[14,5])
    
    # Confusion matrix
    axs = pl.confusion_matrix(axs, pred_clf, true_clf, usedNames)

    # Cross entropy / Confidence metrics
    if nClasses == 2: axs = pl.cross_entropy_metrics(axs, y_true, y_pred[:,1].reshape(-1,1), usedNames)
    else: axs[1] = pl.cross_entropy_multiclass(axs[1], true_clf, y_pred, usedNames)

    pl.show_plot()
    
    v.update({'y_true':y_true, 'y_pred':y_pred, 'true_clf':true_clf, 'pred_clf':pred_clf, 'fig':fig})
    
    return v
  
# Save figure
def save_plot(v, location=False):
    for i in v.keys(): globals()[i] = v[i]
    if location: pl.save_plot(fig, location)
    else: pl.save_plot(fig, 'results/Beirut_UN_nd{}_cls{}{}_neighbours{}{}_std{}_adj{}{}'.format(str(len(nodes)),str(nClasses),str(classesUsed),
                                                                                          str(dataTypes),str(neighbours),str(stdTest),
                                                                                          str(adjacent),str(geoNeighbours)))