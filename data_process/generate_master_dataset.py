import xarray as xr
import pandas as pd
import glob
import numpy as np
import os
from joblib import Parallel, delayed


def find_station_location(station_name):
    station_longitude = meta_airbase.where(station_name, drop=True).Longitude[0]
    station_latitude = meta_airbase.where(station_name, drop=True).Latitude[0]
    station_longitude = longitude.sel(lon=station_longitude, method='nearest')
    station_latitude = latitude.sel(lat=station_latitude, method='nearest')
    return [np.asscalar(station_latitude), np.asscalar(station_longitude)]


def read_era5_time_series_data(variable_name, root_path, spatial_selection_area):
    filenames = f'meteo_interp_{variable_name}_*.nc'
    all_files = glob.glob(os.path.join(root_path, 'data', 'ERA5', region_of_interest, filenames))
    all_datasets = [xr.open_dataset(filename).chunk() for filename in all_files]
    concatenated_dataset = xr.concat(all_datasets, dim='time')
    return concatenated_dataset.sel(**spatial_selection_area).to_dataframe()


def read_no2_time_series_data(root_path, spatial_selection_area):
    all_files = glob.glob(os.path.join(root_path, 'data', 'Sentinelno2', region_of_interest, 'S5P_NO2_*.nc'))
    all_datasets = [xr.open_dataset(filename).chunk() for filename in all_files]
    concatenated_dataset = xr.concat(all_datasets, dim='time')
    return concatenated_dataset.sel(**spatial_selection_area).to_dataframe()


def save_data_from_near_station(filename):
    pickle_name = filename[-21:-4] + '_features.pkl'
    pickle_path = os.path.join(root_directory, 'data', 'pickles', region_of_interest, pickle_name)
    
    if not os.path.isfile(pickle_path):
        station_name = meta_airbase.AirQualityStationEoICode == filename[-21:-14]
        
        try:
            station_location = find_station_location(station_name)
            spatial_selection_area = dict(
                lat=slice(station_location[0] - region_range, station_location[0] + region_range),
                lon=slice(station_location[1] - region_range, station_location[1] + region_range)
            )
            
            u10_data = read_era5_time_series_data('u10', root_directory, spatial_selection_area)
            v10_data = read_era5_time_series_data('v10', root_directory, spatial_selection_area)
            t2m_data = read_era5_time_series_data('t2m', root_directory, spatial_selection_area)
            cdir_data = read_era5_time_series_data('cdir', root_directory, spatial_selection_area)
            tp_data = read_era5_time_series_data('tp', root_directory, spatial_selection_area)
            blh_data = read_era5_time_series_data('blh', root_directory, spatial_selection_area)
            no2_satellite_data = read_no2_time_series_data(root_directory, spatial_selection_area)
            
            meteorological_station_series = pd.merge(u10_data, v10_data, left_index=True, right_index=True, how='inner')
            meteorological_station_series = pd.merge(meteorological_station_series, t2m_data, left_index=True, right_index=True, how='inner')
            meteorological_station_series = pd.merge(meteorological_station_series, cdir_data, left_index=True, right_index=True, how='inner')
            meteorological_station_series = pd.merge(meteorological_station_series, tp_data, left_index=True, right_index=True, how='inner')
            meteorological_station_series = pd.merge(meteorological_station_series, blh_data, left_index=True, right_index=True, how='inner')
            
            no2_time_series = pd.read_csv(filename, usecols=[3, 5])
            no2_time_series['time'] = pd.to_datetime(no2_time_series.DatetimeBegin, utc=True)
            no2_time_series = no2_time_series.set_index('time')
            no2_time_series.index = no2_time_series.index.tz_convert(tz=None)
            no2_time_series = no2_time_series[no2_time_series.index.year == 2018]
            
            merged_dataframe_1 = pd.merge(meteorological_station_series, no2_satellite_data, left_index=True, right_index=True, how='outer')
            final_dataframe = pd.merge(no2_time_series, merged_dataframe_1, left_index=True, right_index=True, how='outer')
            final_dataframe = final_dataframe.drop(columns=['DatetimeBegin'])
            
            time_invariant_data = xr.open_dataset(os.path.join(root_directory, 'input', region_of_interest + '_v2.nc'))
            time_invariant_input = time_invariant_data.sel(**spatial_selection_area).to_dataframe()
            for column_name in list(time_invariant_input):
                final_dataframe[column_name] = np.asscalar(time_invariant_input[column_name])
            
            feature_data = xr.open_dataset(os.path.join(root_directory, 'input', region_of_interest + '_features.nc'))
            feature_input = feature_data.sel(**spatial_selection_area).to_dataframe()
            for column_name in list(feature_input):
                final_dataframe[column_name] = np.asscalar(feature_input[column_name])
            
            dem_feature_data = xr.open_dataset(os.path.join(root_directory, 'input', region_of_interest + '_dem_features.nc'))
            dem_feature_input = dem_feature_data.sel(**spatial_selection_area).to_dataframe()
            for column_name in list(dem_feature_input):
                final_dataframe[column_name] = np.asscalar(dem_feature_input[column_name])
            
            final_dataframe = final_dataframe.reset_index(['lat', 'lon'])
            final_dataframe.to_pickle(pickle_path)
            
        except Exception as e:
            station_code = meta_airbase.where(station_name, drop=True)['AirQualityStationEoICode'][0].values
            print(f"Error processing station {station_code}: file not saved")
    else:
        print(f"File already exists: {filename}")


def main():
    global meta_airbase, longitude, latitude, region_range, region_of_interest, root_directory
    
    root_directory = '.'
    region_of_interest = 'ROI1'
    region_range = 0.001
    
    time_invariant_data = xr.open_dataset(os.path.join(root_directory, 'input', region_of_interest + '_v2.nc'))
    metadata_base = pd.read_csv(os.path.join(root_directory, 'input', 'AIRBASE', 'metadata_AIRBASE.csv'))
    meta_airbase = xr.Dataset.from_dataframe(metadata_base)
    
    airbase_filenames = glob.glob(os.path.join(root_directory, 'data', 'AIRBASE', f'no2_{region_of_interest}', '*.csv'))
    
    latitude = time_invariant_data.lat
    longitude = time_invariant_data.lon
    
    Parallel(n_jobs=-1)(delayed(save_data_from_near_station)(filename) for filename in airbase_filenames)


if __name__ == "__main__":
    main()