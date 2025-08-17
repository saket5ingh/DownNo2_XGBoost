#!/usr/bin/env python3

import xarray as xr
import pandas as pd
import glob
import numpy as np
import os
import pickle
import argparse
import textwrap
from datetime import datetime, timedelta
from joblib import Parallel, delayed


def parse_date(date_string):
    """Parse date string in YYYY-MM-DD or YYYY-DDD format."""
    if len(date_string) == 10:
        return datetime.strptime(date_string, '%Y-%m-%d')
    else:
        return datetime.strptime(date_string, '%Y-%j')


def get_previous_month_last_hours(data_path, year, month):
    """Get data from the last day of the previous month."""
    first_day = f"{year}-{month:02d}-01"
    date = parse_date(first_day) + timedelta(days=-1)
    
    file_path = os.path.join(
        data_path, 'data', 'Sentinelno2', 'omi_filled',
        f'S5P_hourly_NO2_filled_{date.strftime("%Y-%m-%d")}.nc'
    )
    return xr.open_dataset(file_path)


def read_era5_monthly_data(variable, year, month, data_path, spatial_subset):
    """Read ERA5 meteorological data for a specific month."""
    pattern = f'meteo_interp_{variable}_{year}{month:02d}*.nc'
    file_pattern = os.path.join(data_path, 'data', 'ERA5', 'ROI1', pattern)
    
    all_files = glob.glob(file_pattern)
    datasets = [xr.open_dataset(fname).chunk() for fname in all_files]
    combined_data = xr.concat(datasets, dim='time')
    
    return combined_data.sel(**spatial_subset)


def read_no2_monthly_data(data_path, year, month, spatial_subset):
    """Read NO2 hourly data for a specific month."""
    pattern = f'S5P_hourly_NO2_filled_{year}-{month:02d}-*.nc'
    file_pattern = os.path.join(data_path, 'data', 'Sentinelno2', 'omi_filled', pattern)
    
    all_files = sorted(glob.glob(file_pattern))
    datasets = [xr.open_dataset(fname).chunk() for fname in all_files]
    
    additional_data = get_previous_month_last_hours(data_path, year, month)
    datasets.append(additional_data)
    
    combined_data = xr.concat(datasets, dim='time')
    return combined_data.sel(**spatial_subset)


def create_land_use_dummies(dataframe, data_path):
    """Create dummy variables for land use categories."""
    metadata_file = os.path.join(data_path, 'input', 'LUD', 'clc_legend.csv')
    land_use_metadata = pd.read_csv(metadata_file, sep=';')
    
    all_categories = land_use_metadata.CLC_CODE.unique()
    existing_categories = dataframe.lu.unique()
    missing_categories = [cat for cat in all_categories if cat not in existing_categories]
    
    extended_series = dataframe.lu.append(pd.Series(missing_categories))
    extended_series = extended_series.reset_index(drop=True)
    
    dummies = pd.get_dummies(extended_series)
    dummies = dummies.drop(dummies.index[-len(missing_categories):])
    
    major_categories = np.floor(extended_series / 100.)
    major_dummies = pd.get_dummies(major_categories, prefix='lu_major')
    major_dummies = major_dummies.drop(major_dummies.index[-len(missing_categories):])
    
    for code in list(dummies):
        label = land_use_metadata.LABEL3.loc[land_use_metadata.CLC_CODE == code].to_string(index=False)
        label = label.strip().replace(' ', '_')
        dummies = dummies.rename(columns={code: label})
    
    return dummies, major_dummies


def preprocess_data(dataframe, model_id, data_path):
    """Transform and preprocess the input data for model prediction."""
    traffic_mask = np.isnan(dataframe.rld)
    traffic_mask2 = (dataframe.tfv < 0)
    
    dataframe.loc[traffic_mask2, 'tfv'] = np.nan
    dataframe.loc[traffic_mask, 'tfv'] = 0
    dataframe.loc[traffic_mask, 'rld'] = 0
    dataframe.loc[np.isnan(dataframe.lu), 'lu'] = 523
    
    processed_df = pd.DataFrame()
    
    processed_df['Longitude'] = dataframe.lon
    processed_df['Latitude'] = dataframe.lat
    processed_df['Certainty_dist'] = dataframe.certainty_dist
    
    processed_df['10m_u-component_of_wind_speed'] = dataframe.u10
    processed_df['10m_v-component_of_wind_speed'] = dataframe.v10
    processed_df['10m_wind_speed'] = np.linalg.norm([dataframe.u10, dataframe.v10], axis=0)
    processed_df['Temperature_2m'] = dataframe.t2m
    processed_df['Solar_radiation'] = dataframe.cdir
    processed_df['Total_precipitation'] = dataframe.tp
    processed_df['Boundary_layer_height'] = dataframe.blh
    
    processed_df['Digital_Elevation_Map'] = dataframe.dem
    processed_df['Digital_Elevation_Map_wt_400m'] = dataframe.dem_hpf_400m
    processed_df['Digital_Elevation_Map_wt_1km'] = dataframe.dem_hpf_1km
    processed_df['Digital_Elevation_Map_wt_6km'] = dataframe.dem_hpf_6km
    processed_df['Digital_Elevation_Map_wt_25km'] = dataframe.dem_hpf_25km
    processed_df['Digital_Elevation_Map_wt_100km'] = dataframe.dem_hpf_100km
    
    processed_df['Population_density_f'] = np.log1p(dataframe.popf_max)
    processed_df['Population_density_d'] = dataframe.popf_dist
    processed_df['Traffic_volume_f'] = np.log1p(dataframe.tfvf_max)
    processed_df['Traffic_volume_d'] = dataframe.tfvf_dist
    processed_df['Road_length_density_f'] = np.log1p(dataframe.rldf_max)
    processed_df['Road_length_density_d'] = dataframe.rldf_dist
    processed_df['TNO_emission_f'] = np.log1p(dataframe.tnof_max)
    processed_df['TNO_emission_d'] = dataframe.tnof_dist
    processed_df['Land use'] = dataframe.lu
    processed_df['Population_density'] = np.log1p(dataframe['pop'])
    processed_df['Traffic_volume'] = np.log1p(dataframe.tfv)
    processed_df['Road_length_density'] = np.log1p(dataframe.rld)
    
    processed_df['Month'] = dataframe.index.month
    processed_df['Day'] = dataframe.index.day
    processed_df['Hour'] = dataframe.index.hour
    processed_df['weekday'] = (1 * (dataframe.index.dayofweek < 5) + 
                              2 * (dataframe.index.dayofweek == 5) + 
                              3 * (dataframe.index.dayofweek == 6))
    processed_df['doy'] = dataframe.index.dayofyear
    
    processed_df = processed_df.replace([np.inf, -np.inf], np.nan)
    
    scaler_path = os.path.join(data_path, 'models', model_id, 'min_max_scaler_variables.sav')
    scalers = pickle.load(open(scaler_path, 'rb'))
    scaled_data = scalers.transform(processed_df)
    processed_df = pd.DataFrame(scaled_data, columns=list(processed_df))
    processed_df = processed_df.set_index(dataframe.index)
    
    no2_scaler_path = os.path.join(data_path, 'models', model_id, 'pw_scaler_filled_sat.sav')
    no2_scalers = pickle.load(open(no2_scaler_path, 'rb'))
    processed_df['Sentinel5p_no2_fill'] = no2_scalers.transform(
        dataframe.no2_avg_filled.values.reshape(-1, 1)
    )
    
    dummies, major_dummies = create_land_use_dummies(dataframe, data_path)
    dummies = dummies.set_index(dataframe.index)
    major_dummies = major_dummies.set_index(dataframe.index)
    
    final_df = pd.concat([processed_df, dummies], axis=1)
    return final_df.dropna()


def select_model_features(dataframe, model_name='Model0'):
    """Select features based on the specified model configuration."""
    feature_configs = {
        "Model1": ['Digital_Elevation_Map_wt_400m', 'Digital_Elevation_Map_wt_1km',
                  'Digital_Elevation_Map_wt_6km', 'Digital_Elevation_Map_wt_25km',
                  'Digital_Elevation_Map_wt_100km', 'Population_density_f',
                  'Population_density_d', 'Traffic_volume_f', 'Traffic_volume_d',
                  'Road_length_density_f', 'Road_length_density_d',
                  'TNO_emission_f', 'TNO_emission_d', 'doy'],
        
        "Model2": ['Digital_Elevation_Map_wt_400m', 'Digital_Elevation_Map_wt_1km',
                  'Digital_Elevation_Map_wt_6km', 'Digital_Elevation_Map_wt_25km',
                  'Digital_Elevation_Map_wt_100km', 'Population_density',
                  'Traffic_volume', 'Road_length_density', 'doy'],
        
        "Model3": ['Population_density', 'Traffic_volume', 'Road_length_density',
                  'Month', 'Day', 'Hour', 'doy'],
        
        "Model4": ['Population_density', 'Traffic_volume', 'Road_length_density', 'doy'],
        
        "Model5": ['Population_density', 'Traffic_volume', 'Road_length_density',
                  'Day', '10m_u-component_of_wind_speed', '10m_v-component_of_wind_speed'],
        
        "Model0": ['Population_density', 'Traffic_volume', 'Road_length_density',
                  'Day', 'Month', '10m_u-component_of_wind_speed',
                  '10m_v-component_of_wind_speed', 'Longitude', 'Latitude', 'Land use'],
    }
    
    columns_to_drop = feature_configs.get(model_name, [])
    return dataframe.drop(columns=columns_to_drop)


def make_prediction(data_to_predict, model_id, model_name, data_path):
    """Make NO2 concentration predictions using the trained model."""
    def inverse_scale_prediction(prediction):
        scaler_path = os.path.join(data_path, 'models', model_id, 'pw_scaler_0.001percent.sav')
        scalers = pickle.load(open(scaler_path, 'rb'))
        return scalers.inverse_transform(prediction)
    
    model_path = os.path.join(data_path, 'models', model_id, f'XGB_best_{model_name}.sav')
    model = pickle.load(open(model_path, 'rb'))
    
    predictions = pd.DataFrame({'no2': model.predict(data_to_predict)}).set_index(data_to_predict.index)
    predictions = inverse_scale_prediction(predictions)
    
    return pd.DataFrame({'no2': predictions.flatten()}, index=data_to_predict.index).to_xarray()


def process_and_save_prediction(dataset_slice, model_id, model_name, feature_list, data_path):
    """Process a data slice and save the prediction to file."""
    time_obs = dataset_slice.time
    filename = f'ROI1_{model_name}_no2_{str(time_obs.values)[:13]}_lon_{dataset_slice.lon[0].values:5f}_lat_{dataset_slice.lat[0].values:5f}.nc'
    
    output_path = os.path.join(data_path, 'models', model_id, 'Results', filename)
    
    if not os.path.isfile(output_path):
        dataframe = dataset_slice.to_dataframe()
        processed_data = preprocess_data(dataframe.reset_index().set_index('time'), model_id, data_path)
        processed_data = processed_data[feature_list]
        processed_data = processed_data.reset_index().set_index(dataframe.index)
        processed_data = processed_data.drop(columns=['time'])
        
        prediction = make_prediction(processed_data, model_id, model_name, data_path)
        prediction.coords['time'] = time_obs
        prediction = prediction.expand_dims('time')
        prediction.to_netcdf(output_path)
    else:
        print(f'File already exists: {filename}')


def main():
    description = textwrap.dedent("""\
        Generate NO2 concentration maps using trained XGBoost models.
        
        Available Models:
        - Model1: Null model without feature extraction
        - Model2: Spatial features with emission source extraction
        - Model3: Spatial features + wavelet transform of DEM
        - Model4: Dimensionality reduction (selected features)
        - Model0: Default model configuration
        
        The script generates predicted NO2 maps as NetCDF files that can be
        combined using a separate merging script.
    """)
    
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('data_path', type=str, help='Path to the data directory')
    parser.add_argument('model_name', type=str, 
                       choices=['Model0', 'Model1', 'Model2', 'Model3', 'Model4'],
                       help='Model configuration to use')
    parser.add_argument('year', type=int, help='Year to predict (e.g., 2018, 2019, 2020)')
    parser.add_argument('month', type=int, choices=range(1, 13),
                       help='Month to predict (1-12)')
    parser.add_argument('qa_threshold', type=int, choices=[50, 75],
                       help='Quality assurance threshold (50 or 75)')
    
    args = parser.parse_args()
    
    data_path = args.data_path
    model_name = args.model_name
    year = args.year
    month = args.month
    qa_threshold = args.qa_threshold
    
    if not os.path.exists(data_path):
        raise ValueError(f"Data path does not exist: {data_path}")
    
    lat_range = np.arange(42, 48.5, 1)
    lon_range = np.arange(6, 12.5, 1)
    
    model_id = f'Full_features{qa_threshold}'
    
    model_path = os.path.join(data_path, 'models', model_id, f'XGB_best_{model_name}.sav')
    model = pickle.load(open(model_path, 'rb'))
    feature_list = model.get_booster().feature_names
    
    chunks = {'time': 1, 'lon': -1, 'lat': -1}
    
    results_dir = os.path.join(data_path, 'models', model_id, 'Results')
    os.makedirs(results_dir, exist_ok=True)
    
    for lon_idx in range(6):
        for lat_idx in range(6):
            print(f"Processing grid cell: lon_idx={lon_idx}, lat_idx={lat_idx}")
            
            spatial_subset = {
                'lat': slice(lat_range[lat_idx], lat_range[lat_idx + 1]),
                'lon': slice(lon_range[lon_idx], lon_range[lon_idx + 1])
            }
            
            time_invariant_data = xr.open_dataset(os.path.join(data_path, 'input', 'ROI1_v2.nc'))
            feature_data = xr.open_dataset(os.path.join(data_path, 'input', 'ROI1_features_v2.nc'))
            dem_feature_data = xr.open_dataset(os.path.join(data_path, 'input', 'ROI1_dem_features_dmey.nc'))
            
            static_data = xr.merge([time_invariant_data, feature_data, dem_feature_data], 
                                 join='outer').sel(**spatial_subset)
            
            met_u10 = read_era5_monthly_data('u10', year, month, data_path, spatial_subset)
            met_v10 = read_era5_monthly_data('v10', year, month, data_path, spatial_subset)
            met_t2m = read_era5_monthly_data('t2m', year, month, data_path, spatial_subset)
            met_cdir = read_era5_monthly_data('cdir', year, month, data_path, spatial_subset)
            met_tp = read_era5_monthly_data('tp', year, month, data_path, spatial_subset)
            met_blh = read_era5_monthly_data('blh', year, month, data_path, spatial_subset)
            
            met_data = xr.merge([met_u10, met_v10, met_t2m, met_cdir, met_tp, met_blh], join='inner')
            _, unique_indices = np.unique(met_data['time'], return_index=True)
            
            no2_data = read_no2_monthly_data(data_path, year, month, spatial_subset)
            no2_data = no2_data.reindex_like(static_data, method='nearest')
            _, unique_indices = np.unique(no2_data['time'], return_index=True)
            no2_data = no2_data.isel(time=unique_indices)
            
            combined_dataset = xr.merge([met_data, no2_data, static_data], join='inner')
            combined_dataset = combined_dataset.chunk(chunks=chunks)
            combined_dataset = combined_dataset.transpose('time', 'lat', 'lon')
            combined_dataset = combined_dataset.sortby('time')
            
            Parallel(n_jobs=-1)(
                delayed(process_and_save_prediction)(
                    combined_dataset.isel(time=i), model_id, model_name, feature_list, data_path
                ) for i in range(len(combined_dataset.time))
            )
    
    print("Prediction generation completed!")


if __name__ == '__main__':
    main()