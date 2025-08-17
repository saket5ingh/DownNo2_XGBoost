import os
import xarray as xr
import pandas as pd
import glob
from joblib import Parallel, delayed

def read_NO2_time_series_data(root, ROI):
    all_files = glob.glob(os.path.join(root, 'data', 'Sentinelno2', ROI, 'S5P_NO2_*.nc'))
    all_dsets = [xr.open_dataset(fname).chunk() for fname in all_files]
    return xr.concat(all_dsets, dim='time').sortby('time')

def temporal_interpolation_satellite_data(ds_part, root, ROI):
    target_file = os.path.join(root, 'data', 'Sentinelno2', ROI, 'S5P_hourly_NO2_' + str(ds_part.time[0].values)[0:10] + '.nc')
    if not os.path.isfile(target_file):
        temptt = pd.date_range(ds_part.time[0].values, ds_part.time[-1].values, freq='H')
        temptt = xr.DataArray(temptt, dims='time')
        ds_part = ds_part.reindex_like(temptt).chunk(chunks={'time': -1, 'lon': 256, 'lat': 256})
        ds_part = ds_part.interpolate_na(dim='time', limit=24)
        ds_part[0:-1, :, :].to_netcdf(target_file)
    else:
        print('file exists :', os.path.basename(target_file))

def temporal_interpolation_satellite_data_end(ds_part, root, ROI):
    target_file = os.path.join(root, 'data', 'Sentinelno2', ROI, 'S5P_hourly_NO2_' + str(ds_part.time[0].values)[0:10] + '.nc')
    if not os.path.isfile(target_file):
        temptt = pd.date_range(ds_part.time[0].values, ds_part.time[-1].values, freq='H')
        temptt = xr.DataArray(temptt, dims='time')
        ds_part = ds_part.reindex_like(temptt).chunk(chunks={'time': -1, 'lon': 256, 'lat': 256})
        ds_part = ds_part.interpolate_na(dim='time', limit=24)
        ds_part.to_netcdf(target_file)
    else:
        print('file exists :', os.path.basename(target_file))

def main():
    root = input("Enter the root folder where Sentinel data is stored: ").strip()
    if not os.path.isdir(root):
        print("Invalid root folder path. Exiting.")
        return
    ROI = input("Enter the region of interest (e.g., ROI1, ROI2): ").strip()
    mfds = read_NO2_time_series_data(root, ROI)
    Parallel(n_jobs=-1)(delayed(temporal_interpolation_satellite_data)(mfds.no2[t:(t+2), :, :], root, ROI) for t in range(1, (len(mfds.time)-2), 1))
    temporal_interpolation_satellite_data_end(mfds.no2[-2:, :, :], root, ROI)

if __name__ == '__main__':
    main()
