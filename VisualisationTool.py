import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import cartopy.crs as ccrs
import xarray as xr
import cartopy.feature as cf
import seaborn as sns

filepath1 = '/Users/kenzatazi/Downloads/yields.csv'
filepath2 = '/Users/kenzatazi/Downloads/predictions_with_countries.csv'
filepath3 = '/Users/kenzatazi/Downloads/head_of_soils_recommendations_MGM-2.csv'    

sns.set(style="white", context="talk")

def prediction_formatting(filepath2, filepath3):
    
    df =  pd.read_csv(filepath3)
    df_hist = df[['x','y','maize_a_2010', 'iso3']]

    df = pd.read_csv(filepath2)
    df_new= df[['lat', 'lon', '2040_mean', '2040_std', '2025_mean', '2025_std', '2020_mean', '2020_std']]

    df_hist['2040_change'] = df_new['2040_mean']/df_hist['maize_a_2010'] - 1
    df_hist['2025_change'] = df_new['2025_mean']/df_hist['maize_a_2010'] - 1
    df_hist['2020_change'] = df_new['2020_mean']/df_hist['maize_a_2010'] - 1

    return df_hist

def static_worldmap(df_raw, year='2040'):
    """ Produces static plot of world yield predictions """

    # create dataframe of relevant variables
    df = df_raw[['x','y', year+'_change','iso3']]

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
    df_values = df[['x','y', year+'_change']]
    df_pv = df_values.pivot(index='y', columns='x')
    df_pv = df_pv.droplevel(0, axis=1)
    da = xr.DataArray(data=df_pv)

    # plot
    plt.figure(figsize=(12,5))
    ax = plt.subplot(projection=ccrs.PlateCarree())
    da.plot.pcolormesh('x', 'y', ax=ax, vmin=-1, vmax=1, extend='both', cmap='RdBu_r',
                       cbar_kwargs={'fraction': 0.019, 'pad': 0.10, 'format': tck.PercentFormatter(xmax=1.0) }) #'label': '%'
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


def static_countrymap(df_raw, year='2040', iso3='USA', RCP='8.5'):
    """ Produces static plot of a given country's yield predictions """
    
    coords = {'USA': [-135, -65, 22, 55], 'CHN': [71, 140, 15, 55],
              'BRA': [-80, -30, -40, 8]}

    # create dataframe of relevant variables
    df = df_raw[['x','y', year+'_change','iso3']]
    country_df = df[df['iso3'] == iso3]
    country_change = country_df[year+'_change'].mean()

    # convert dataframe to data array
    df_values = country_df[['x','y', year+'_change']]
    df_pv = df_values.pivot(index='y', columns='x')
    df_pv = df_pv.droplevel(0, axis=1)
    da = xr.DataArray(data=df_pv)

    # plot
    plt.figure(figsize=(12,5))
    ax = plt.subplot(projection=ccrs.PlateCarree())
    da.plot.pcolormesh('x', 'y', ax=ax, vmin=-1, vmax=1, extend='both', cmap='RdBu_r',
                       cbar_kwargs={'fraction': 0.019, 'pad': 0.10,'format': tck.PercentFormatter(xmax=1.0)})
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


def yield_vs_time(df_raw, year):
    """ Return a matplotlib graph of yield as a function of time """

    # Seperate data into classes

    # Plot
    plt.figure(figsize=(12,5))
    plt.title('Yield change as a function of time')
    plt.xlabel('Year')
    plt.ylabel('Yield change')


def feature_importance(df_raw):
    """ returns plot of feature importance """
    
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


