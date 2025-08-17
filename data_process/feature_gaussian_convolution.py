import os
import numpy as np
import xarray as xr
import scipy.ndimage as ndi


def def_bbox(ROI):
    if ROI == 'ROI1':
        return [6, 12, 42, 48]  # Alpine (around Switzerland, northern Italy, alpine area)
    else:
        return [2, 8, 48, 54]   # Benelux (northern Europe)


def gaussian_convolution(root, ROI):
    
    tinvData = xr.open_dataset(os.path.join(root, 'input', f'{ROI}_v2.nc'))
    lat = tinvData.lat
    lon = tinvData.lon

    ds = xr.Dataset()

    
    for i in range(10):
        sd = 2 ** i

        
        x = tinvData.pop.values
        xg = ndi.gaussian_filter(x, sd)
        ds[f'pop_{sd}'] = xr.DataArray(xg, coords={'lat': lat, 'lon': lon}, dims=['lat', 'lon'])

        
        x = tinvData.rld.values
        x[np.isnan(x)] = 0
        xg = ndi.gaussian_filter(x, sd)
        ds[f'rld_{sd}'] = xr.DataArray(xg, coords={'lat': lat, 'lon': lon}, dims=['lat', 'lon'])

        
        x = tinvData.tfv.values
        x[np.isnan(x)] = 0
        xg = ndi.gaussian_filter(x, sd)
        ds[f'tfv_{sd}'] = xr.DataArray(xg, coords={'lat': lat, 'lon': lon}, dims=['lat', 'lon'])

    
    out_path = os.path.join(root, 'data', f'{ROI}_tinvData_gaussian.nc')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ds.to_netcdf(out_path)
    print(f"Saved Gaussian convolved data: {out_path}")


def main():
    root = input("Enter the root folder path: ").strip()
    if not os.path.isdir(root):
        print("Invalid root folder path. Exiting.")
        return

    ROI = input("Enter the Region of Interest (ROI1 or ROI2): ").strip()
    gaussian_convolution(root, ROI)


if __name__ == "__main__":
    main()
