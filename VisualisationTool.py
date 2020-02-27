import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import cartopy.crs as ccrs
import xarray as xr

filepath = '/Users/kenzatazi/Downloads/yields.csv'
year = '2040'

def static_worldmap(filepath, year):
    """ Produces static plot of yield predictions """

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
    da.plot.pcolormesh('x', 'y', ax=ax, cbar_kwargs={'fraction': 0.019, 'pad': 0.10, 'format': tck.PercentFormatter(xmax=1.0)}) #'label': '%'
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
