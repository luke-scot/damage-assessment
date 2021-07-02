import numpy as np
import ipywidgets as ipw
import demo_plotting as pl
import demo_imports as ip
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
    beirutDefs = {'gtFile':'./data/beirutDamages.shp',
                 'crs':'EPSG:4326',
                 'labelColumn':'decision',
                 'map':True,
                 'mapOption':False,
                 'lat':33.893,
                 'lon':35.512,
                 'zoom':14,
                 'dataTypes':['High resolution imagery','Interferometry'],
                 'dataFiles':["data/highRes/20JUL31_HR_LatLon.tif","data/highRes/20AUG05_HR_LatLon.tif","./data/beirutPrePreExplosionIfg.tif","./data/beirutPrePostExplosionIfg.tif"],
                 'polyBounds':[35.512407312180784, 33.892243658425194, 35.52800151162857, 33.90124240848098],
                 'colors':['green','yellow','cyan','red','maroon'],
                 }
    # Houston
    houstonDefs = {'gtFile':'./data/2018IEEE_Contest/Phase2/TrainingGT/2018_IEEE_GRSS_DFC_GT_TR.tif',
                  'crs':'EPSG:26915',
                  'labelColumn':'class',
                  'map':False,
                  'dataTypes':['Hyperspectral imagery','LiDAR','High resolution imagery'],
                  'dataFiles':[None, './data/2018IEEE_Contest/Phase2/FullHSIDataset/20170218_UH_CASI_S4_NAD83.pix',None,'./data/2018IEEE_Contest/Phase2/Lidar GeoTiff Rasters/Intensity_C3/UH17_GI3F051.tif', None,'./data/2018IEEE_Contest/Phase2/Final RGB HR Imagery']
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
    v.update({'bxNClasses':bxNClasses,'bxNodes':bxNodes,'bxEdges':bxEdges,'bxAdjacent':bxAdjacent,'unique':unique,'labels': labels})
    return v  