import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import cartopy.crs as ccrs
import xarray as xr
import cartopy.feature as cf
import seaborn as sns


filepath = '/Users/kenzatazi/Downloads/Predictions for Years_ [2040, 2025, 2020].csv'
    

sns.set(style="white", context="talk")

def prediction_formatting(filepath):
    """ Returns a dataframe with columns for map plotting. """
    
    df =  pd.read_csv(filepath)

    df['2040_change'] = df['2040_mean']/df['maize_a_2010'] - 1
    df['2025_change'] = df['2025_mean']/df['maize_a_2010'] - 1
    df['2020_change'] = df['2020_mean']/df['maize_a_2010'] - 1

    return df


def change_worldmap(df_raw, year='2040'):
    """ Returns plot of world yield changes from 2010 to a given year (2020, 2025 or 2040).s"""

    # create dataframe of relevant variables
    df = df_raw[['lon','lat', year+'_change','iso3']]

    # calculate mean changes for global, LIFDC and US
    global_change = df[ year+'_change'].mean()
    lifdc_df = df[df['iso3'].isin(['AFG', 'BGD', 'BEN', 'BGD', 'BDI', 'CMR', 'CAF',
                    'TCD', 'COG', 'CIV', 'PRK', 'COD', 'DJI', 'ERI', 'ETH', 'GMB', 'GHA',
                    'GNB', 'HTI', 'IND', 'KEN', 'KGZ', 'LSO', 'LBR', 'MDG', 'MWI', 'MLI',
                    'MRT', 'MOZ', 'NPL', 'NIC', 'NER', 'RWA', 'STP', 'SEN', 'SLE', 'SOM',
                    'SLP', 'SSD', 'SDN', 'SYR', 'TJK', 'TGO', 'UGA', 'TZA', 'UZB', 'VNM',
                    'YEM', 'ZWE'])]
    lifdc_change = lifdc_df[year+'_change'].mean()
    us_df = df[df['iso3'] == 'USA']
    us_change = us_df[ year+'_change'].mean()

    # text for box
    t1 = 'Yield change:'
    t2 = 'USA= {:.2%}'.format(us_change)
    t3 = 'LIFDC = {:.2%}'.format(lifdc_change)
    t4 = 'Global= {:.2%}'.format(global_change)

    # convert dataframe to data array
    df_values = df[['lon','lat', year+'_change']]
    df_pv = df_values.pivot(index='lat', columns='lon')
    df_pv = df_pv.droplevel(0, axis=1)
    da = xr.DataArray(data=df_pv)

    # plot
    plt.figure(figsize=(12,5))
    ax = plt.subplot(projection=ccrs.PlateCarree())
    da.plot.pcolormesh('lon', 'lat', ax=ax, vmax=1, extend='both', cmap='RdBu_r',
                       cbar_kwargs={'fraction': 0.019,'pad': 0.10, 'format': tck.PercentFormatter(xmax=1.0) }) #'label': '%'
    gl = ax.gridlines(draw_labels=True)
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}
    ax.coastlines(resolution='50m')
    ax.set_extent([-160, 180, -60, 85])
    ax.set_title('Maize Yield Change ' + year + '\n', size='x-large')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect("equal")
    
    plt.text(-170,-50, t1 + '\n' + t2 + '\n' + t3 + '\n'+ t4, fontsize=10,
             bbox=dict(facecolor='white', edgecolor='grey', pad=10.0))
    plt.show()

def mean_worldmap(df_raw, year='2040'):
    """ Returns worldmap of mean yield predictions for given year (2020, 2025 or 2040). """

    # create dataframe of relevant variables
    df = df_raw[['lon','lat', year+'_mean','iso3']]
    df['yield'] = df[year+'_mean']/1000

    # calculate mean changes for global, LIFDC and US
    global_change = df['yield'].mean()
    lifdc_df = df[df['iso3'].isin(['AFG', 'BGD', 'BEN', 'BGD', 'BDI', 'CMR', 'CAF',
                    'TCD', 'COG', 'CIV', 'PRK', 'COD', 'DJI', 'ERI', 'ETH', 'GMB', 'GHA',
                    'GNB', 'HTI', 'IND', 'KEN', 'KGZ', 'LSO', 'LBR', 'MDG', 'MWI', 'MLI',
                    'MRT', 'MOZ', 'NPL', 'NIC', 'NER', 'RWA', 'STP', 'SEN', 'SLE', 'SOM',
                    'SLP', 'SSD', 'SDN', 'SYR', 'TJK', 'TGO', 'UGA', 'TZA', 'UZB', 'VNM',
                    'YEM', 'ZWE'])]
    lifdc_change = lifdc_df['yield'].mean()
    us_df = df[df['iso3'] == 'USA']
    us_change = us_df['yield'].mean()

    # text for box
    t1 = 'Yield:'
    t2 = 'USA= {:.2f} (ton/ha)'.format(us_change)
    t3 = 'LIFDC = {:.2f} (ton/ha)'.format(lifdc_change)
    t4 = 'Global= {:.2f} (ton/ha)'.format(global_change)

    # convert dataframe to data array
    df_values = df[['lon','lat', year+'_mean']]
    df_pv = df_values.pivot(index='lon', columns='lat')
    df_pv = df_pv.droplevel(0, axis=1)
    da = xr.DataArray(data=df_pv)

    # plot
    plt.figure(figsize=(12,5))
    ax = plt.subplot(projection=ccrs.PlateCarree())
    da.plot.pcolormesh('lon', 'lat', ax=ax, cmap='magma_r',
                       cbar_kwargs={'fraction': 0.01, 'pad': 0.10) #'label': '%'
    ax.gridlines(draw_labels=True)
    ax.coastlines(resolution='50m')
    ax.set_extent([-160, 180, -60, 85])
    ax.set_title('Maize Yield ' + year + '\n', size='x-large')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect("equal")
    
    plt.text(-170,-50, t1 + '\n' + t2 + '\n' + t3 + '\n'+ t4, fontsize=10,
             bbox=dict(facecolor='white', edgecolor='grey', pad=10.0))
    plt.show()

def std_worldmap(df_raw, year='2040'):
    """ Returns world map of standard deviations in yield predictions for 2020, 2025 or 2040. """

    # create dataframe of relevant variables
    df = df_raw[['lon','lat', year+'_std','iso3']]
    df['yield'] = df[year+'_std']/1000

    # calculate mean changes for global, LIFDC and US
    global_change = df['yield'].mean()
    lifdc_df = df[df['iso3'].isin(['AFG', 'BGD', 'BEN', 'BGD', 'BDI', 'CMR', 'CAF',
                    'TCD', 'COG', 'CIV', 'PRK', 'COD', 'DJI', 'ERI', 'ETH', 'GMB', 'GHA',
                    'GNB', 'HTI', 'IND', 'KEN', 'KGZ', 'LSO', 'LBR', 'MDG', 'MWI', 'MLI',
                    'MRT', 'MOZ', 'NPL', 'NIC', 'NER', 'RWA', 'STP', 'SEN', 'SLE', 'SOM',
                    'SLP', 'SSD', 'SDN', 'SYR', 'TJK', 'TGO', 'UGA', 'TZA', 'UZB', 'VNM',
                    'YEM', 'ZWE'])]
    lifdc_change = lifdc_df['yield'].mean()
    us_df = df[df['iso3'] == 'USA']
    us_change = us_df['yield'].mean()

    # text for box
    t1 = 'Yield standard deviation:'
    t2 = 'USA= {:.2f} ton/ha'.format(us_change)
    t3 = 'LIFDC = {:.2f} ton/ha'.format(lifdc_change)
    t4 = 'Global= {:.2f} ton/ha'.format(global_change)

    # convert dataframe to data array
    df_values = df[['lon','lat','yield']]
    df_pv = df_values.pivot(index='lon', columns='lat')
    df_pv = df_pv.droplevel(0, axis=1)
    da = xr.DataArray(data=df_pv)

    # plot
    plt.figure(figsize=(12,5))
    ax = plt.subplot(projection=ccrs.PlateCarree())
    da.plot.pcolormesh('lon', 'lat', ax=ax, cbar_kwargs={'fraction': 0.019, 'pad': 0.05},
                       cmap= 'magma_r')
    gl = ax.gridlines(draw_labels=True)
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}
    ax.coastlines(resolution='50m')
    ax.set_extent([-160, 180, -60, 85])
    ax.set_title('Maize Yield Standard Deviation ' + year + '\n', size='x-large')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect("equal")
    
    plt.text(-170,-50, t1 + '\n' + t2 + '\n' + t3 + '\n'+ t4, fontsize=10,
             bbox=dict(facecolor='white', edgecolor='grey', pad=10.0))
    plt.show()


def change_countrymap(df_raw, year='2040', iso3='USA'):
    """ Produces map of a given country's change in yield from 2010 to 2020, 2025 or 2040. """
    
    coords = {'USA': [-135, -65, 22, 50], 'CHN': [71, 140, 10, 50],
              'BRA': [-80, -30, -45, 8]}

    # create dataframe of relevant variables
    df = df_raw[['lon','lat', year+'_change','iso3']]
    country_df = df[df['iso3'] == iso3]
    country_change = country_df[year+'_change'].mean()

    # convert dataframe to data array
    df_values = country_df[['lon','lat', year+'_change']]
    df_pv = df_values.pivot(index='lat', columns='lon')
    df_pv = df_pv.droplevel(0, axis=1)
    da = xr.DataArray(data=df_pv)

    # plot
    plt.figure(figsize=(12,5))
    ax = plt.subplot(projection=ccrs.PlateCarree())
    da.plot.pcolormesh('lon', 'lat', ax=ax, vmin=-1, vmax=1, extend='both', cmap='RdBu_r',
                       cbar_kwargs={'fraction': 0.019, 'pad': 0.10,'format': tck.PercentFormatter(xmax=1.0)})
    ax.gridlines(draw_labels=True)
    ax.coastlines(resolution='50m')
    ax.add_feature(cf.BORDERS)
    ax.set_extent(coords[iso3])  
    ax.set_title('Maize Yield Change ' + year + '\n', size='xx-large')
    ax.set_aspect("equal")
    
    t1 = 'National yield change: {:.2%}'.format(country_change)
    plt.text((coords[iso3])[0]+5, (coords[iso3])[2]+5, t1, fontsize=12, 
             bbox=dict(facecolor='white', edgecolor='grey', pad=10.0))
    plt.show()

def mean_countrymap(df_raw, year='2040', iso3='USA'):
    """ Produces map of a given country's mean yield predictions for 2020, 2025 or 2040. """
    
    coords = {'USA': [-135, -65, 22, 50], 'CHN': [71, 140, 10, 50],
              'BRA': [-80, -30, -45, 8]}

    # create dataframe of relevant variables
    df = df_raw[['lon','lat', year+'_mean','iso3']]
    df['yield'] = df[year+'_mean']/1000

    country_df = df[df['iso3'] == iso3]
    country_change = country_df['yield'].mean()

    # convert dataframe to data array
    df_values = country_df[['lon','lat', 'yield']]
    df_pv = df_values.pivot(index='lat', columns='lon')
    df_pv = df_pv.droplevel(0, axis=1)
    da = xr.DataArray(data=df_pv)

    # plot
    plt.figure(figsize=(12,5))
    ax = plt.subplot(projection=ccrs.PlateCarree())
    da.plot.pcolormesh('lon', 'lat', ax=ax, cmap= 'magma_r',
                       cbar_kwargs={'fraction': 0.019, 'pad': 0.10})
    ax.gridlines(draw_labels=True)
    ax.coastlines(resolution='50m')
    ax.add_feature(cf.BORDERS)
    ax.set_extent(coords[iso3])  
    ax.set_title('Maize Yield ' + year + '\n', size='x-large')
    ax.set_aspect("equal")
    
    t1 = 'National yield: {:.2f} ton/ha'.format(country_change)
    plt.text((coords[iso3])[0]+5, (coords[iso3])[2]+5, t1, fontsize=10, 
             bbox=dict(facecolor='white', edgecolor='grey', pad=10.0))
    plt.show()

def std_countrymap(df_raw, year='2040', iso3='USA'):
    """ Produces map of a given country's standard deviations in yield predictions for 2020, 2025 or 2040. """
    
    coords = {'USA': [-135, -65, 22, 50], 'CHN': [71, 140, 10, 50],
              'BRA': [-80, -30, -45, 8]}

    # create dataframe of relevant variables
    df = df_raw[['lon','lat', year+'_std','iso3']]
    df['yield'] = df[year+'_std']/1000

    country_df = df[df['iso3'] == iso3]
    country_change = country_df['yield'].mean()

    # convert dataframe to data array
    df_values = country_df[['lon','lat', 'yield']]
    df_pv = df_values.pivot(index='lat', columns='lon')
    df_pv = df_pv.droplevel(0, axis=1)
    da = xr.DataArray(data=df_pv)

    # plot
    plt.figure(figsize=(12,5))
    ax = plt.subplot(projection=ccrs.PlateCarree())
    da.plot.pcolormesh('lon', 'lat', ax=ax, cmap= 'magma_r',
                       cbar_kwargs={'fraction': 0.019, 'pad': 0.10})
    ax.gridlines(draw_labels=True)
    ax.coastlines(resolution='50m')
    ax.add_feature(cf.BORDERS)
    ax.set_extent(coords[iso3])  
    ax.set_title('Maize Yield Standard Deviation ' + year + '\n', size='x-large')
    ax.set_aspect("equal")
    
    t1 = 'National standard deviation: {:.2f} ton/ha'.format(country_change)
    plt.text((coords[iso3])[0]+5, (coords[iso3])[2]+5, t1, fontsize=10, 
             bbox=dict(facecolor='white', edgecolor='grey', pad=10.0))
    plt.show()


def yield_vs_time(filepath):
    """ Returns violin graph of yield distribution as a function of time. """

    # Seperate data into classes
    df= pd.read_csv(filepath)

    df_2020 = pd.concat([df['2p6_2019_predict'], df['2p6_2020_predict'], df['2p6_2021_predict'],
                         df['4p5_2019_predict'], df['4p5_2020_predict'], df['4p5_2021_predict'],
                         df['6p0_2019_predict'], df['6p0_2020_predict'], df['6p0_2021_predict'],
                         df['8p5_2019_predict'], df['8p5_2020_predict'], df['8p5_2021_predict']],
                         ignore_index=True, axis=0)

    df_2025 = pd.concat([df['2p6_2024_predict'], df['2p6_2025_predict'], df['2p6_2026_predict'],
                         df['4p5_2024_predict'], df['4p5_2025_predict'], df['4p5_2026_predict'],
                         df['6p0_2024_predict'], df['6p0_2025_predict'], df['6p0_2026_predict'],
                         df['8p5_2024_predict'], df['8p5_2025_predict'], df['8p5_2026_predict']],
                         ignore_index=True, axis=0)

    df_2040 = pd.concat([df['2p6_2039_predict'], df['2p6_2040_predict'], df['2p6_2042_predict'],
                         df['4p5_2039_predict'], df['4p5_2040_predict'], df['4p5_2042_predict'],
                         df['6p0_2039_predict'], df['6p0_2040_predict'], df['6p0_2042_predict'],
                         df['8p5_2039_predict'], df['8p5_2040_predict'], df['8p5_2042_predict']],
                         ignore_index=True, axis=0)
    
    df_violin = pd.concat([df['maize_a_2010'], df_2020, df_2025, df_2040], axis=1)
    df_violin.columns = ['2010', '2020', '2025', '2040']
    df_violin = df_violin.div(1000)

    # Plot

    plt.figure()
    plt.title('Yield as a function of time \n')
    plt.xlabel('Year')
    plt.ylabel('Yield (ton/ha) \n')
    plt.grid(which= 'major', axis='y')
    sns.violinplot(data=df_violin, palette='Blues', inner="quart")
    plt.show()

def feature_importance(df_raw):
    """ Dummy function to create feature importance graph. """
    
    # fake dataframe 
    df = pd.DataFrame({'Feature1': 80, 'Feature2': 50, 'Feature3': 44, 
                       'Feature4': 13, 'Feature5': 2, 'Feature6': 0.09})

    # df =  pd.read_csv(filepath)
    # names = df['Features'].values
    # importance = df['Importance'].values

    sns.barplot(data= df, palette="rocket", orient='h')
    plt.title('Feature Importance')
    plt.xlabel('Importance (%)')

    # could make this more interesting by including the spread and error bar
