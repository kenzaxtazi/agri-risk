import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import json

from bokeh.io import output_notebook, show, output_file, save
from bokeh.plotting import figure
from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar,  NumeralTickFormatter
from bokeh.palettes import brewer


# creating the dataframe for peaches and nectarines in 2017
filepath = '/Users/kenzatazi/Downloads/Production_Crops_E_All_Data/Production_Crops_E_All_Data.csv'
df =  pd.read_csv(filepath, encoding='latin-1')
df1 = df[['Area', 'Item', 'Element', 'Unit', 'Y2017']]
df2 = df1[df1['Item']=='Peaches and nectarines']
df3 = df2[df2['Element']=='Yield']

vmax = df3['Y2017'].max()


# country shapes 
shapefile = '/Users/kenzatazi/Downloads/ne_50m_admin_0_countries/ne_50m_admin_0_countries.shp'
gdf = gpd.read_file(shapefile)[['ADMIN', 'ADM0_A3', 'geometry']]
gdf.columns = ['country', 'country_code', 'geometry']
gdf = gdf.drop(gdf.index[159])  # remove Antarctica 

merged = gdf.merge(df3, left_on = 'country', right_on = 'Area', how = 'left')
#merged.fillna({'Y2017': 0}, inplace = True)

#Read data to json
merged_json = json.loads(merged.to_json())

#Convert to str like object
json_data = json.dumps(merged_json)

#Input GeoJSON source that contains features for plotting.
geosource = GeoJSONDataSource(geojson = json_data)

#Define a sequential multi-hue color palette.
palette = brewer['Oranges'][5]

#Reverse color order so that dark blue is highest obesity.
palette = palette[::-1]

#Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors.
color_mapper = LinearColorMapper(palette = palette, low = 0, high = vmax, nan_color = '#d9d9d9')

#Define custom tick labels for color bar.
tick_labels = {'0': '0', str(vmax): "{:.2e}".format(vmax)}

#Create color bar
format_tick = NumeralTickFormatter(format='0 a')
color_bar = ColorBar(color_mapper=color_mapper, label_standoff=5, width=20, height=500,
border_line_color=None, location = (0,0), orientation = 'vertical', formatter=format_tick)

#Create figure object.
p = figure(title = 'Peach and nectarine yields 2017 (hg/ha)', plot_height = 600 , plot_width = 950, toolbar_location = None)
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None

#Add patch renderer to figure. 
p.patches('xs','ys', source = geosource, fill_color = {'field' :'Y2017', 'transform' : color_mapper},
          line_color = 'black', line_width = 0.25, fill_alpha = 1)

#Specify figure layout.
p.add_layout(color_bar, 'right')

#Display figure.
output_file('foo.html')
save(p)
show(p)

