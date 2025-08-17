import os
import xarray as xr
import numpy as np
import scipy.interpolate as intp
from datetime import datetime, timedelta
import argparse
import textwrap
import glob

def parse_date(s):
    if len(s) == 10:
        return datetime.strptime(s, '%Y-%m-%d')
    else:
        return datetime.strptime(s, '%Y-%j')

def iter_dates(start, stop):
    current = start
    while current <= stop:
        yield current
        current += timedelta(days=1)
        
def ROI_mask_for_regridding(lat, lon):
    thres = 0.05
    maskROI = [lon.min()-thres,
               lon.max()+thres,
               lat.min()-thres,
               lat.max()+thres]        
    return maskROI

def reindex_sentinel(start, stop, ROI, root='.'):
    filenc = input("Enter the full path to land usage data (.nc file): ").strip()
    if not os.path.isfile(filenc):
        print("Invalid file path for land usage data. Exiting.")
        return

    dsROI = xr.open_dataset(filenc)
    lat = dsROI.lat
    lon = dsROI.lon
    lonm, latm = np.meshgrid(lon, lat) 
    maskROI = ROI_mask_for_regridding(lat, lon)
    datestr = '%Y%m%d'
    
    for date in iter_dates(start, stop):
        print(date.strftime(datestr))
        no2 = []
        pathname = os.path.join(root, 'Sentinelno2', 'S5P_OFFL_L2__NO2____' + date.strftime(datestr) + '*.nc')
        filenames = glob.glob(pathname)

        for f in filenames:
            ds = xr.open_dataset(f, group='PRODUCT').rename({'longitude': 'lon', 'latitude': 'lat'})
            bbROI = (ds.lon > maskROI[0]) & (ds.lon < maskROI[1]) & (ds.lat > maskROI[2]) & (ds.lat < maskROI[3]) & (ds.qa_value > 0.5) & (ds.nitrogendioxide_tropospheric_column > 0)
            dsROI = ds.where(bbROI, drop=True)
            if dsROI.lon.size != 0:
                x1 = dsROI.lon[0, :, :].values.ravel()
                y1 = dsROI.lat[0, :, :].values.ravel()
                no2_original = dsROI.nitrogendioxide_tropospheric_column[0, :, :].values.ravel()
                points1 = np.column_stack((x1, y1))
                regridded_no2 = intp.griddata(points1, no2_original, (lonm, latm), method='nearest')
                regridded_no2_1 = intp.griddata(points1, no2_original, (lonm, latm), method='linear')
                regridded_no2[np.isnan(regridded_no2_1)] = np.nan
                no2_o = xr.DataArray(regridded_no2, coords=[lon, lat], dims=['lon', 'lat'])
                no2.append(no2_o)
                
        if no2:
            no2 = xr.concat(no2, dim='time')
            no2 = no2.chunk(chunks={'lon': 256, 'lat': 256})   
            no2 = no2.mean(dim='time', skipna=True) 
            ds = no2.to_dataset(name='no2')
            newfilename = 's5p_no2_' + ROI + '_' + date.strftime(datestr) + '.nc'
            target = os.path.join(root, 'Sentinelno2', newfilename)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            ds.to_netcdf(target)

def main():
    root = input("Enter the root folder where Sentinel data is stored: ").strip()
    if not os.path.isdir(root):
        print("Invalid root folder path. Exiting.")
        return

    description = textwrap.dedent("""\
        Python script for reindexing Sentinel 5p data (Level 2).
    """)

    parser = argparse.ArgumentParser(description=description, epilog='',
            formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('starttime', type=str, help='start date (YYYY-mm-dd or YYYY-jjj)')
    parser.add_argument('stoptime', type=str, help='stop date (YYYY-mm-dd or YYYY-jjj)')
    parser.add_argument('ROI', type=str, help='region of interest (ROI1, ROI2, etc.)')

    args = parser.parse_args()

    start = parse_date(args.starttime)
    stop = parse_date(args.stoptime)
    ROI = args.ROI

    reindex_sentinel(start, stop, ROI=ROI, root=root)

if __name__ == '__main__':
    main()
