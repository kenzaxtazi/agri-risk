import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
import seaborn as sns
import cartopy.crs as ccrs
import xarray as xr
import cartopy.feature as cf

filepath1 = '/Users/kenzatazi/Downloads/head_of_soils_recommendations_MGM-2.csv'
filepath2 = '/Users/kenzatazi/Downloads/climate_monthly_seasonal_2005_2010_with_spatial_attributes_filtered.csv'


def historical_yield(filepath1, year):
    """ Returns the global map of maize yield """
    
    # create dataframe of relevant variables
    df_raw =  pd.read_csv(filepath1)
    df = df_raw[['x','y','maize_a_' + year, 'iso3']]
    df['maize_a_' + year] = df['maize_a_' + year]/1000 

    # calculate mean for global, LIFDC and US
    global_mean = df['maize_a_' + year].mean()
    lifdc_df = df[df['iso3'].isin(['AFG', 'BGD', 'BEN', 'BGD', 'BDI', 'CMR', 'CAF',
                    'TCD', 'COG', 'CIV', 'PRK', 'COD', 'DJI', 'ERI', 'ETH', 'GMB', 'GHA',
                    'GNB', 'HTI', 'IND', 'KEN', 'KGZ', 'LSO', 'LBR', 'MDG', 'MWI', 'MLI',
                    'MRT', 'MOZ', 'NPL', 'NIC', 'NER', 'RWA', 'STP', 'SEN', 'SLE', 'SOM',
                    'SLP', 'SSD', 'SDN', 'SYR', 'TJK', 'TGO', 'UGA', 'TZA', 'UZB', 'VNM',
                    'YEM', 'ZWE'])]
    lifdc_mean = lifdc_df['maize_a_' + year].mean()
    us_df = df[df['iso3'] == 'USA']
    us_mean = us_df['maize_a_' + year].mean()


    # text for box
    t1 = 'Average yield :'
    t2 = 'USA= {:.2f} ton/ha'.format(us_mean)
    t3 = 'LIFDC = {:.2f} ton/ha'.format(lifdc_mean)
    t4 = 'Global= {:.2f} ton/ha'.format(global_mean)

    # to DataArray
    df_values = df[['x','y','maize_a_' + year]]
    df_pv = df_values.pivot(index='y', columns='x')
    df_pv = df_pv.droplevel(0, axis=1)
    da = xr.DataArray(data=df_pv)

    # plot
    plt.figure(figsize=(12,5))
    ax = plt.subplot(projection=ccrs.PlateCarree())
    da.plot.pcolormesh('x', 'y', ax=ax, cmap='magma_r', vmin=0,
                       cbar_kwargs={'fraction': 0.019, 'pad': 0.10,'format': '%.2f'})
    ax.gridlines(draw_labels=True)
    ax.coastlines(resolution='50m')
    ax.set_extent([-160, 180, -60, 85])
    ax.set_title('Maize Yield ' + year + '\n', size='xx-large')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    plt.text(-170,-50, t1 + '\n' + t2 + '\n' + t3 + '\n'+ t4, fontsize=10,
             bbox=dict(facecolor='white', edgecolor='grey', pad=10.0))
    plt.show()


def agroclimatic_indicators(filepath1):
    """ returns histograms of the different indicators for 2010"""
    # Data (look at spread in different places over one year)

    df_raw =  pd.read_csv(filepath1)
    area_weights(df_raw, 'y')
    w = (df_raw['area weights'].values).flatten()
    
    # Frost days (days)
    FD = (df_raw[['FD-01-05-2010', 'FD-02-05-2010', 'FD-03-05-2010', 'FD-04-05-2010',
                  'FD-05-05-2010', 'FD-06-05-2010', 'FD-07-05-2010', 'FD-08-05-2010', 
                  'FD-09-05-2010', 'FD-10-05-2010', 'FD-11-05-2010', 'FD-12-05-2010', 
                  'FD-01-15-2010', 'FD-02-15-2010', 'FD-03-15-2010', 'FD-04-15-2010',
                  'FD-05-15-2010', 'FD-06-15-2010', 'FD-07-15-2010', 'FD-08-15-2010', 
                  'FD-09-15-2010', 'FD-10-15-2010']].values).flatten()

    # Biological Effective Degree Days (°C)
    BEDD =  (df_raw[['BEDD-01-05-2010', 'BEDD-02-05-2010', 'BEDD-03-05-2010',
                     'BEDD-04-05-2010', 'BEDD-05-05-2010', 'BEDD-06-05-2010',
                     'BEDD-07-05-2010', 'BEDD-08-05-2010', 'BEDD-09-05-2010',
                     'BEDD-10-05-2010', 'BEDD-11-05-2010', 'BEDD-12-05-2010',
                     'BEDD-01-15-2010', 'BEDD-02-15-2010', 'BEDD-03-15-2010',
                     'BEDD-04-15-2010', 'BEDD-05-15-2010', 'BEDD-06-15-2010',
                     'BEDD-07-15-2010', 'BEDD-08-15-2010', 'BEDD-09-15-2010',
                     'BEDD-10-15-2010', 'BEDD-11-15-2010', 'BEDD-12-15-2010', 
                     'BEDD-01-25-2010', 'BEDD-02-25-2010', 'BEDD-03-25-2010',
                     'BEDD-04-25-2010', 'BEDD-05-25-2010', 'BEDD-06-25-2010',
                     'BEDD-07-25-2010', 'BEDD-08-25-2010', 'BEDD-09-25-2010', 
                     'BEDD-10-25-2010', 'BEDD-11-25-2010', 'BEDD-12-25-2010']].values).flatten()
  
    WSDI = (df_raw[['WSDI-Q1-2010', 'WSDI-Q2-2010', 'WSDI-Q3-2010', 'WSDI-Q4-2010']].values).flatten()
    CSDI = (df_raw[['CSDI-Q1-2010', 'CSDI-Q2-2010', 'CSDI-Q3-2010', 'CSDI-Q4-2010']].values).flatten()                
   
    WW = (df_raw[['WW-Q1-2010', 'WW-Q2-2010', 'WW-Q3-2010', 'WW-Q4-2010']].values).flatten()


    CWD = (df_raw[['CWD-Q1-2010', 'CWD-Q2-2010', 'CWD-Q3-2010', 'CWD-Q4-2010']].values).flatten()
    CFD = (df_raw[['CFD-Q1-2010', 'CFD-Q2-2010', 'CFD-Q3-2010', 'CFD-Q4-2010']].values).flatten()
    CDD = (df_raw[['CDD-Q1-2010', 'CDD-Q2-2010', 'CDD-Q3-2010', 'CDD-Q4-2010']].values).flatten()

    w_FD = np.tile(w, 22)
    w_quaterly = np.tile(w, 4)
    w_10day = np.tile(w, 36)

    #  Plots
    fig1, axs1 = plt.subplots(2, 1)
    axs1[0].hist(FD, weights=w_FD, bins=10, density=True, label='Frost Days')
    axs1[0].legend(facecolor='white')
    axs1[1].hist(WW, weights=w_quaterly, bins=50, density=True, label='Warm and Wet Days')
    axs1[1].legend(facecolor='white')
    axs1[1].set_xlabel('Days')


    fig2, axs2 = plt.subplots(3, 1)
    axs2[0].hist(CDD, weights=w_quaterly, bins=92, density=True, label='Maximum Consecutive Dry Days')
    axs2[0].legend(facecolor='white')
    axs2[1].hist(CFD, weights=w_quaterly, bins=92, density=True, label='Maximum Consecutive Frost Days')
    axs2[1].legend(facecolor='white')
    axs2[2].hist(CWD, weights=w_quaterly, bins=92, density=True, label='Maximum Consecutive Wet Days')
    axs2[2].legend(facecolor='white')
    axs2[2].set_xlabel('Days')

    fig3, axs3 = plt.subplots(1, 1)
    axs3.hist(BEDD, weights=w_10day, bins=110, density=True, label='Biological Effective Degree Days')
    axs3.legend(facecolor='white')
    axs3.set_xlabel('°C')

    fig4, axs4 = plt.subplots(2, 1)
    axs4[0].hist(CSDI, weights=w_quaterly, bins=40, density=True, label='Cold Spell Duration Index')
    axs4[0].legend(facecolor='white')
    axs4[1].hist(WSDI, weights=w_quaterly, bins=44, density=True, label='Warm Spell Duration Index')
    axs4[1].legend(facecolor='white')
    axs4[1].set_xlabel('Days')
    plt.show()


def soil_types(filepath1):
    
    df_raw =  pd.read_csv(filepath1)
    area_weights(df_raw, 'y')

    df = df_raw['soil_types']
    plt.figure()
    plt.ylabel('Count')
    plt.xlabel('Soil type')
    df.hist(weights= df_raw['area weights'], bins=107, align='mid', density=True, rwidth=0.75)
    plt.show()


def irrigation(filepath2):
    """ Returns map of irrigated and waterfed maize """
    
    # create dataframe of relevant variables
    df_raw =  pd.read_csv(filepath2)
    df_values = df_raw[['lon','lat','irrigation']]

    df_pv = df_values.pivot(index='lat', columns='lon')
    df_pv = df_pv.droplevel(0, axis=1)
    da = xr.DataArray(data=df_pv)

    # plot
    plt.figure(figsize=(12,5))
    ax = plt.subplot(projection=ccrs.PlateCarree())
    da.plot.pcolormesh('lon', 'lat', ax=ax, cbar_kwargs={'fraction': 0.019, 'pad': 0.10, 
                                                         'format': tck.PercentFormatter(xmax=100.0)}) 
    ax.gridlines(draw_labels=True)
    ax.coastlines(resolution='50m')
    ax.set_extent([-160, 180, -60, 85])
    ax.set_title('Irrigation \n ', size='xx-large')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    plt.show()


def elevation_slope(filepath2):
    """ Returns histograms of elevation and slope """

    df_raw =  pd.read_csv(filepath2)

    
    area_weights(df_raw, 'lat')
    w = (df_raw['area weights'].values).flatten()

    elevation = (df_raw['elevation'].values).flatten()
    slope = (df_raw['slope'].values).flatten()
    df_values = df_raw[['lon','lat','elevation']]

    df_pv = df_values.pivot(index='lat', columns='lon')
    df_pv = df_pv.droplevel(0, axis=1)
    da = xr.DataArray(data=df_pv)

    #  Plots
    fig, axs = plt.subplots(2, 1)
    axs[0].hist(elevation, weights=w, bins=50, density=True, label='Elevation')
    axs[0].legend(facecolor='white')
    axs[0].set_xlabel('m')

    axs[1].hist(slope, weights=w, bins=50, density=True, label= 'Slope')
    axs[1].legend(facecolor='white')
    axs[1].set_xlabel('deg')

    plt.figure(figsize=(12,5))
    ax = plt.subplot(projection=ccrs.PlateCarree())
    da.plot.pcolormesh('lon', 'lat', ax=ax, cbar_kwargs={'fraction': 0.019, 'pad': 0.10, 
                                                         'format': tck.PercentFormatter(xmax=100.0)}) 
    ax.gridlines(draw_labels=True)
    ax.coastlines(resolution='50m')
    ax.set_extent([-160, 180, -60, 85])
    ax.set_title('Elevation \n ', size='xx-large')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    plt.show()


def climate_zones(filepath1):

    """ bar chart with climate zones of haversted areas """

    names = ['Tropics', 'Subtropics\n(summer\nrainfall)',
             'Subtropics\n(winter\nrainfall)', 'Temperate\n(oceanic)',
             'Temperate\n(sub-\ncontinental)', 'Temperate\n(continental)',
             'Boreal\n(oceanic)', 'Boreal\n(sub-\ncontinental)',
             'Boreal \n(continental)', 'Arctic']
    
    df_raw =  pd.read_csv(filepath1)
    df = df_raw[['climate_zones', 'y']]
    df_clean = df[df['climate_zones'] >= 0]
    area_weights(df_clean, 'y')

    hist, bin_edges = np.histogram(df_clean['climate_zones'], weights=df_clean['area weights'], density=True)
    print(hist)

    plt.figure()    
    plt.bar(names, hist)
    plt.title('Climate zones')
    plt.show()


def area_weights(df, lat_column):
    """ returns area weights for a given latitude """

    theta1 = (abs(df[lat_column]) - 0.041667) * np.pi /180.0
    theta2 = (abs(df[lat_column]) + 0.041667) * np.pi /180.0

    df['area weights'] = (abs(np.cos(theta1)) - abs(np.cos(theta2)))


def soil_grouping(filepath1):
    
    df_raw =  pd.read_csv(filepath1)
    area_weights(df_raw, 'y')

    df = df_raw[['soil_types', 'area weights']]

    df = df.replace(to_replace=np.arange(1,15), value='Cambisol')
    df = df.replace(to_replace=np.arange(15,19), value='Chernozem')
    df = df.replace(to_replace=np.arange(19,22), value='Podzoluvisol')
    df = df.replace(to_replace=22, value='Rendzima')
    df = df.replace(to_replace= np.arange(23,29), value='Ferrasol')
    df = df.replace(to_replace= np.arange(29,36), value='Gleysol')
    df = df.replace(to_replace= np.arange(36,40), value='Phaozem')
    df = df.replace(to_replace=40, value='Lithosol')
    df = df.replace(to_replace= np.arange(41,45), value='Fluvisol')
    df = df.replace(to_replace= np.arange(45,48), value='Kastanozem')
    df = df.replace(to_replace= np.arange(48,56), value='Luvisol')
    df = df.replace(to_replace= np.arange(56,58), value='Greyzem')
    df = df.replace(to_replace= np.arange(58,61), value='Nitosol')
    df = df.replace(to_replace= np.arange(61,64), value='Histosol')
    df = df.replace(to_replace= np.arange(64,69), value='Podzol')
    df = df.replace(to_replace= np.arange(69,74), value='Arenosol')
    df = df.replace(to_replace= np.arange(74,78), value='Regosol')
    df = df.replace(to_replace= np.arange(78,81), value='Solonetz')
    df = df.replace(to_replace= np.arange(81,85), value='Andosol')
    df = df.replace(to_replace=85, value='Ranker')
    df = df.replace(to_replace= np.arange(86,88), value='Vertisol')
    df = df.replace(to_replace= np.arange(88,94), value='Planosol')
    df = df.replace(to_replace= np.arange(94,98), value='Xerosol')
    df = df.replace(to_replace= np.arange(98,103), value='Yermosol')
    df = df.replace(to_replace= np.arange(103,107), value='Solonchak')
    df = df.replace(to_replace=107, value='Ice')
    df = df.replace(to_replace=108, value='NA')

    df1 = df.groupby(['soil_types']).sum()
    total = df1['area weights'].sum()
    df1['fraction'] =  df1['area weights']/total
    df1.plot.bar(y='fraction', legend=False)
    plt.ylabel('Area fraction')
    plt.xlabel('Soil type')
    plt.show()



