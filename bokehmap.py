import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import json

from bokeh.io import output_notebook, output_file, save, curdoc
from bokeh.plotting import figure, show
from bokeh.resources import CDN
from bokeh.embed import file_html
from bokeh.models import Slider, HoverTool, Select
from bokeh.layouts import widgetbox, row, column
from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar, NumeralTickFormatter, ColumnDataSource, CustomJS
from bokeh.palettes import brewer


# creating the dataframe for peaches and nectarines in 2017
filepath = '/Users/kenzatazi/Downloads/Production_Crops_E_All_Data/Production_Crops_E_All_Data_NOFLAG.csv'
df =  pd.read_csv(filepath, encoding='latin-1')
df1 = df.drop(['Area Code', 'Item Code', 'Element Code', 'Unit'], axis=1)
df2 = df1[df1['Element']=='Yield']
df3 = pd.melt(df2, id_vars=['Area', 'Item'], value_vars=['Y1961', 'Y1962', 'Y1963', 'Y1964', 'Y1965', 'Y1966', 'Y1967', 'Y1968',
                                                       'Y1969', 'Y1970', 'Y1971', 'Y1972', 'Y1973', 'Y1974', 'Y1975', 'Y1976',
                                                       'Y1977', 'Y1978', 'Y1979', 'Y1980', 'Y1981', 'Y1982', 'Y1983', 'Y1984',
                                                       'Y1985', 'Y1986', 'Y1987', 'Y1988', 'Y1989', 'Y1990', 'Y1991', 'Y1992', 
                                                       'Y1993', 'Y1994', 'Y1995', 'Y1996', 'Y1997', 'Y1998', 'Y1999', 'Y2000', 
                                                       'Y2001', 'Y2002', 'Y2003', 'Y2004', 'Y2005', 'Y2006', 'Y2007', 'Y2008', 
                                                       'Y2009', 'Y2010', 'Y2011', 'Y2012', 'Y2013', 'Y2014', 'Y2015', 'Y2016', 
                                                       'Y2017'], var_name='Year', value_name='Yield')
df3['Yield'] = df3['Yield'].div(1e4)                                                    
df3 = df3[(df3['Item']== 'Wheat') | (df3['Item']== 'Soybeans') | (df3['Item']== 'Maize') | (df3['Item']== 'Rice, paddy')]
vmax = df3['Yield'].max()


# country shapes 
shapefile = '/Users/kenzatazi/Downloads/ne_50m_admin_0_countries/ne_50m_admin_0_countries.shp'
gdf = gpd.read_file(shapefile)[['ADMIN', 'ADM0_A3', 'geometry']]
gdf.columns = ['country', 'country_code', 'geometry']
gdf = gdf.drop(gdf.index[159])  # remove Antarctica 

merged = gdf.merge(df3, left_on = 'country', right_on = 'Area', how = 'left')
#merged.fillna({'Y2017': 0}, inplace = True)

#Read data to json
merged_json = json.loads(merged.to_json())

def json_data(yr, cr, df3):
    df_yr = df3[df3['Year'] == yr]
    df_cr = df_yr [df_yr['Item'] == cr]
    merged = gdf.merge(df_cr, left_on= 'country', right_on = 'Area', how = 'left')
    merged_json = json.loads(merged.to_json())
    json_data = json.dumps(merged_json)
    return json_data

#Input GeoJSON source that contains features for plotting.
geosource = GeoJSONDataSource(geojson = json_data('Y2017', 'Wheat', df3))

#Define a sequential multi-hue color palette.
palette = brewer['Oranges'][7]

#Reverse color order 
palette = palette[::-1]

#Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors.
color_mapper = LinearColorMapper(palette = palette, low = 0, high = vmax, nan_color = '#d9d9d9')

#Add hover tool
hover = HoverTool(tooltips = [ ('Country','@Area'),('Yield (hg/ha)', '@Yield')])

#Define custom tick labels for color bar.
tick_labels = {'0': '0', '9.19': '9.19', '13.8':'13.8', '18.4':'18.4', '23.0':'23.0',
               '27.6':'27.6', '32.2':'32.2', str(vmax): str(vmax)}

#Create color bar
#format_tick = NumeralTickFormatter(format='0 a')
color_bar = ColorBar(color_mapper= color_mapper, label_standoff=10, width=20, height=500,
border_line_color=None, location = (0,0), orientation = 'vertical', major_label_overrides = tick_labels)

#Create figure object.
p = figure(plot_height = 500 , plot_width = 950, toolbar_location = None, tools = [hover])
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None

#Add patch renderer to figure. 
p.patches('xs','ys', source = geosource, fill_color = {'field' :'Yield', 'transform' : color_mapper},
          line_color = 'black', line_width = 0.25, fill_alpha = 1)

#Specify figure layout.
p.add_layout(color_bar, 'right')

# Define the callback function: update_plot
def update_plot(attr, old, new):
    yr = slider.value  # input for slider
    cr = select.value  # input for select box
    print(cr, yr)
    new_data = json_data(yr, cr, df3)
    geosource.geojson = new_data

# Make a slider object: slider 
slider = Slider(title = 'Year', start = 1961 , end = 2017, step = 10, value = 2017)
#slider.on_change('value', update_plot)
slider.js_on_change('value', CustomJS.from_py_func(update_plot))

# Make a selection object: select
select = Select(title='Crop', value='Crop', options=['Wheat', 'Soybeans', 'Maize', 'Rice, paddy'])
select.js_on_change('value', CustomJS.from_py_func(update_plot))

# Make a column layout of widgetbox(slider) and plot, and add it to the current document
layout = column(p, widgetbox(slider), widgetbox(select))
curdoc().add_root(layout)

#Display figure.
output_file('foo.html')
html = file_html(p, CDN, "my plot")
save(p)
show(layout)


