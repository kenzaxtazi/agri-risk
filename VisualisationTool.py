import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import cartopy.crs as ccrs
import xarray as xr
import cartopy.feature as cf

filepath = '/Users/kenzatazi/Downloads/yields.csv'
year = '2040'

def static_worldmap(filepath, year):
    """ Produces static plot of world yield predictions """

    # create dataframe of relevant variables
    df_raw =  pd.read_csv(filepath)
    df = df_raw[['x','y','maiz_percent_change','iso3_2005']]

    # calculate mean changes for global, LIFDC and UK
    global_change = df['maiz_percent_change'].mean()
    lifdc_df = df[df['iso3_2005'].isin(['AFG', 'BGD', 'BEN', 'BGD', 'BDI', 'CMR', 'CAF',
                    'TCD', 'COG', 'CIV', 'PRK', 'COD', 'DJI', 'ERI', 'ETH', 'GMB', 'GHA',
                    'GNB', 'HTI', 'IND', 'KEN', 'KGZ', 'LSO', 'LBR', 'MDG', 'MWI', 'MLI',
                    'MRT', 'MOZ', 'NPL', 'NIC', 'NER', 'RWA', 'STP', 'SEN', 'SLE', 'SOM',
                    'SLP', 'SSD', 'SDN', 'SYR', 'TJK', 'TGO', 'UGA', 'TZA', 'UZB', 'VNM',
                    'YEM', 'ZWE'])]
    lifdc_change = lifdc_df['maiz_percent_change'].mean()
    us_df = df[df['iso3_2005'] == 'USA']
    us_change = us_df['maiz_percent_change'].mean()

    # text for box
    t1 = 'Yield change:'
    t2 = 'USA= {:.2%}'.format(us_change)
    t3 = 'LIFDC = {:.2%}'.format(lifdc_change)
    t4 = 'Global= {:.2%}'.format(global_change)

    # convert dataframe to data array
    df_values = df[['x','y','maiz_percent_change']]
    df_pv = df_values.pivot(index='y', columns='x')
    df_pv = df_pv.droplevel(0, axis=1)
    da = xr.DataArray(data=df_pv)

    # plot
    plt.figure(figsize=(12,5))
    ax = plt.subplot(projection=ccrs.PlateCarree())
    da.plot.pcolormesh('x', 'y', ax=ax, cbar_kwargs={'fraction': 0.019, 'pad': 0.10, 
                                                     'format': tck.PercentFormatter(xmax=1.0)}) #'label': '%'
    ax.gridlines(draw_labels=True)
    ax.coastlines(resolution='50m')
    ax.set_extent([-160, 180, -60, 85])
    ax.set_title('Maize Yield Change ' + year + '\n', size='xx-large')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect("equal")
    
    plt.text(-170,-50, t1 + '\n' + t2 + '\n' + t3 + '\n'+ t4, fontsize=10,
             bbox=dict(facecolor='white', edgecolor='grey', pad=10.0))
    plt.show()

def static_countrymap(filepath, year, iso3='USA'):
    """ Produces static plot of a given country's yield predictions """
    
    coords = {'USA': [-135, -65, 22, 55], 'CHN': [71, 140, 15, 55],
              'BRA': [-80, -30, -40, 8]}

    # create dataframe of relevant variables
    df_raw =  pd.read_csv(filepath)
    df = df_raw[['x','y','maiz_percent_change','iso3_2005']]
    country_df = df[df['iso3_2005'] == iso3]
    country_change = country_df['maiz_percent_change'].mean()

    # convert dataframe to data array
    df_values = country_df[['x','y','maiz_percent_change']]
    df_pv = df_values.pivot(index='y', columns='x')
    df_pv = df_pv.droplevel(0, axis=1)
    da = xr.DataArray(data=df_pv)

    # plot
    plt.figure(figsize=(12,5))
    ax = plt.subplot(projection=ccrs.PlateCarree())
    da.plot.pcolormesh('x', 'y', ax=ax, cbar_kwargs={'fraction': 0.019, 'pad': 0.10, 
                                                     'format': tck.PercentFormatter(xmax=1.0)})
    ax.gridlines(draw_labels=True)
    ax.coastlines(resolution='50m')
    ax.add_feature(cf.BORDERS)
    ax.set_extent(coords[iso3])  
    ax.set_title('Maize Yield Change ' + year + '\n', size='xx-large')
    ax.set_aspect("equal")
    
    t1 = 'National yield change: {:.2%}'.format(country_change)
    plt.text((coords[iso3])[0]+5, (coords[iso3])[2]+5, t1, fontsize=10, 
             bbox=dict(facecolor='white', edgecolor='grey', pad=10.0))
    plt.show()


def yield_vs_time(filepath, year):
    """ Return a matplotlib graph of yield as a function of time """

    # Seperate data into classes

    # Plot
    plt.figure(figsize=(12,5))
    plt.title('Yield change as a function of time')
    plt.xlabel('Year')
    plt.ylabel('Yield change')


def feature_importance(filepath):
    """ returns plot of feature importance """

    df =  pd.read_csv(filepath)
    names = df['Features'].values
    importance = df['Importance'].values
    fig = plt.figure()
    plt.title('Feature Importance')
    plt.barh(importance, tick_label=names)

    # could make this more interesting by including the spread


