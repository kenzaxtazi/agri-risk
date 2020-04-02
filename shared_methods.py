# This file contains methods that are used in different scripts of the repo

import pandas as pd
import xarray as xr
import numpy as np
import rasterio

def compute_ten_day_feature(feature, file_location, year, dataset, interpolation_method='linear', in_place_2010=False):
    """
    This method computes the ten day features for a given data set.

    Args:
        feature (str): The String for the agriclimatic indicator. For example: TXx
        file_location (str): Location of the agrimclimatic indicator file.
        year (str): The year you are considering. For example 2015.
        dataset (pandas dataframe): The data set containing latitude and longitude coordinates.
        interpolation_method (str): Method of interpolation. Defaults to linear.
        in_place_2010: Whether to name the column 2010 or the year you are lookin at.

    Returns:
        Updated dataset.
    """
    days = ['05', '15', '25']
    months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    lats_ = xr.DataArray(list(dataset['lat'].values), dims='z')
    lons_ = xr.DataArray(list(dataset['lon'].values), dims='z')
    with xr.open_dataset(file_location) as ds:
        feature_data = ds.load()
        for day in days:
            for month in months:
                time = f'{year}-{month}-{day}'
                timed = feature_data.sel(time=time).squeeze()
                feature_interpolated = timed.interp(lat=lats_, lon=lons_, method=interpolation_method)
                dataset[f'{feature}-{month}-{day}-{year}'] = getattr(feature_interpolated, feature)
    all_ten_days = []
    for month in months:
        for day in days:
            all_ten_days.append(f'{feature}-{month}-{day}-{year}')
        if in_place_2010:
            feature_name = f'{feature}-{month}-2010'
        else:
            feature_name = f'{feature}-{month}-{year}'
        dataset[feature_name] = dataset[f'{feature}-{month}-05-{year}'] + dataset[f'{feature}-{month}-15-{year}'] + dataset[f'{feature}-{month}-25-{year}']
    dataset = dataset.drop(columns=all_ten_days)

    return dataset

def compute_seasonal_feature(feature, file_location, year, dataset, interpolation_method='linear', in_place_2010=False):
    """
    This method computes the ten day features for a given data set.

    Args:
        feature (str): The String for the agriclimatic indicator. For example: TXx
        file_location (str): Location of the agrimclimatic indicator file.
        year (str): The year you are considering. For example 2015.
        dataset (pandas dataframe): The data set containing latitude and longitude coordinates.
        interpolation_method (str): Method of interpolation. Defaults to linear.
        in_place_2010: Whether to name the column 2010 or the year you are lookin at.

    Returns:
        Updated dataset.
    """
    lats = xr.DataArray(list(dataset['lat'].values), dims='z')
    lons = xr.DataArray(list(dataset['lon'].values), dims='z')
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    quarter_time_mapping = {
        'Q1': '01-16',
        'Q2': '04-16',
        'Q3': '07-16',
        'Q4': '10-16'
    }
    with xr.open_dataset(file_location) as ds:
        feature_data = ds.load()

    for quarter in quarters:
        if in_place_2010:
            feature_name = f'{feature}-{quarter}-2010'
        else:
            feature_name = f'{feature}-{quarter}-{year}'
        time = f'{year}-{quarter_time_mapping[quarter]}'
        timed = feature_data.sel(time=time).squeeze()
        feature_interpolated = timed.interp(lat=lats, lon=lons, method=interpolation_method)
        dataset[feature_name] = getattr(feature_interpolated, feature)
    return dataset


def add_degree_split(degree_separation, data_set):
    """
    This method creates a column in a dataframe used to split the data
    into groups based on their degree separation. It splits by slices on longitude
    seperated by a specified amount of degrees

    Args:
        degree_separation (int): The degree_separation between slices of
                                 longitude.

        data_set (pandas dataframe): The data set that contains longitude
                                     information titled as 'lon'

    Returns:
        An updated pandas dataframe.
    """
    lon_slices = int(360/degree_separation) + 1
    lon_range = np.linspace(-180,180,lon_slices)

    def select_bin(row):
        for idx, edge in enumerate(lon_range[:-1]):
            if row['lon'] > edge and row['lon'] < lon_range[idx + 1]:
                return idx % 4

    data_set['degree_split'] = data_set.apply(select_bin, axis=1)

    return data_set
