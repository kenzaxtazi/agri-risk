import gdal
import pandas as pd
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit


## CROP DATA

# open file
filepath1 = 'wheat_YieldPerHectare.tif'
ds = gdal.Open(filepath1)
band = ds.GetRasterBand(1)
yields = band.ReadAsArray() # crop yield of 5*5 min grid



## AGROCLIMATE INDINCATORS

# Cumulitive Dry Days 2000
filepath2 = '/Users/kenzatazi/Downloads/dataset-sis-agroclimatic-indicators-b0d1eedd-e315-46a5-820a-89ae873089c6/CDD_C3S-glob-agric_WFDEI_hist_season_19810101-20101231_v1.nc'
cum_dry_days = xr.open_dataset(filepath2)
CDD_winter = (cum_dry_days.CDD).sel(time = '2000-01-16T00:00:00.000000000')
CDD_spring = (cum_dry_days.CDD).sel(time = '2000-04-16T00:00:00.000000000')
CDD_summer = (cum_dry_days.CDD).sel(time = '2000-07-16T00:00:00.000000000')
CDD_autumn = (cum_dry_days.CDD).sel(time = '2000-10-16T00:00:00.000000000')
CDD_sum = CDD_winter + CDD_spring + CDD_summer + CDD_autumn

# Cumulitive Frost Days 2000
filepath3 = '/Users/kenzatazi/Downloads/dataset-sis-agroclimatic-indicators-b0d1eedd-e315-46a5-820a-89ae873089c6/CFD_C3S-glob-agric_WFDEI_hist_season_19810101-20101231_v1.nc'
cum_frost_days = xr.open_dataset(filepath3)
CFD_winter = (cum_frost_days.CFD).sel(time = '2000-01-16T00:00:00.000000000')
CFD_spring = (cum_frost_days.CFD).sel(time = '2000-04-16T00:00:00.000000000')
CFD_summer = (cum_frost_days.CFD).sel(time = '2000-07-16T00:00:00.000000000')
CFD_autumn = (cum_frost_days.CFD).sel(time = '2000-10-16T00:00:00.000000000')
CFD_sum = CFD_winter + CFD_spring + CFD_summer + CFD_autumn

# Cumulitive Wet Days 2000
filepath4 = '/Users/kenzatazi/Downloads/dataset-sis-agroclimatic-indicators-b0d1eedd-e315-46a5-820a-89ae873089c6/CWD_C3S-glob-agric_WFDEI_hist_season_19810101-20101231_v1.nc'
cum_wet_days = xr.open_dataset(filepath4)
CWD_winter = (cum_wet_days.CWD).sel(time = '2000-01-16T00:00:00.000000000')
CWD_spring = (cum_wet_days.CWD).sel(time = '2000-04-16T00:00:00.000000000')
CWD_summer = (cum_wet_days.CWD).sel(time = '2000-07-16T00:00:00.000000000')
CWD_autumn = (cum_wet_days.CWD).sel(time = '2000-10-16T00:00:00.000000000')
CWD_sum = CWD_winter + CWD_spring + CWD_summer + CWD_autumn

# Resampling
lons_new = np.linspace(cum_dry_days.lon.min(), cum_dry_days.lon.max(), 4320)
lats_new = np.linspace(cum_dry_days.lat.min(), cum_dry_days.lat.max(), 2160)
CWD = CWD_sum.interp(coords={'lat':lats_new, 'lon':lons_new}, method='nearest')
CFD = CFD_sum.interp(coords={'lat':lats_new, 'lon':lons_new}, method='nearest')
CDD = CDD_sum.interp(coords={'lat':lats_new, 'lon':lons_new}, method='nearest')

# to a DataFrame
df = CDD.to_dataframe()
CWD_df = CWD.to_dataframe()
CFD_df = CFD.to_dataframe()
df['CWD'] = CWD_df['CWD']
df['CFD'] = CFD_df['CFD']
df['Yield'] = yields.flatten()

# remove NaN
df_clean = df.dropna()
df_final = df_clean.reset_index()

# Divide into stratified validation and training data
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, val_index in split.split(df_final, df_final['lat'],df_final['lon']):
    train_df = df_final.loc[train_index]
    val_df = df_final.loc[val_index]






