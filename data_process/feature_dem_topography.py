import os
import numpy as np
import xarray as xr
import pywt
from dask.diagnostics import ProgressBar

# functions
def get_new_index_for_wt(ROI):
    if ROI == 'ROI1':
        latt = np.linspace(41.95, 48.05, 5000)
        lont = np.linspace(5.95, 12.05, 5000)
    else:
        latt = np.linspace(47.95, 54.05, 5000)
        lont = np.linspace(1.95, 8.05, 5000)
    return latt, lont

def get_target_index(root, ROI):
    tinvData = xr.open_dataset(os.path.join(root, 'input', ROI + '_v2.nc'))
    lon = tinvData.lon
    lat = tinvData.lat
    return lat, lon

def process_dem(root, ROI):
    demROI = xr.open_dataset(os.path.join(root, 'input', 'eu_dem_' + ROI + '.nc'))
    dem = demROI.Band1
    latt, lont = get_new_index_for_wt(ROI)
    # reindex DEM to even number grid for wavelet transform
    dem = dem.reindex(lat=latt, lon=lont, method='nearest')
    dem.chunk()
    resultswt = []

    for level in range(10):
        coeffs = pywt.wavedec2(dem, wavelet='db2', level=level)
        coeffs[0] = np.zeros_like(coeffs[0])
        filtered = pywt.waverec2(coeffs, 'db2')
        resultswt.append(filtered)

    ds = xr.Dataset({
        'dem_hpf_1': (['lat', 'lon'], resultswt[9]),  # ~100km scale
        'dem_hpf_2': (['lat', 'lon'], resultswt[7]),  # ~25km scale
        'dem_hpf_3': (['lat', 'lon'], resultswt[5]),  # ~6km scale
        'dem_hpf_4': (['lat', 'lon'], resultswt[3])   # ~1.5km scale
    }, coords={'lon': lont, 'lat': latt})

    chunks = {'lon': 256, 'lat': 256}
    lat, lon = get_target_index(root, ROI)
    ds = ds.reindex(lat=lat, lon=lon, method='nearest')
    ds = ds.chunk(chunks=chunks)

    out_file = os.path.join(root, 'input', ROI + '_dem_features.nc')
    with ProgressBar():
        ds.to_netcdf(out_file)
    print(f"Saved DEM features for {ROI} to {out_file}")

def main():
    root = input("Enter the root folder where DEM input data is stored: ").strip()
    if not os.path.isdir(root):
        print("Invalid root folder path. Exiting.")
        return

    ROIlist = ['ROI1', 'ROI2']
    for ROI in ROIlist:
        print(f"Processing {ROI}...")
        process_dem(root, ROI)

if __name__ == '__main__':
    main()
