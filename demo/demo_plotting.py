# Plotting
import random
import ipyleaflet as ipl
import matplotlib.pyplot as plt

"""ipyleaflet plotting functions"""
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
