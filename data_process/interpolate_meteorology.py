import os
import numpy as np
import xarray as xr
from joblib import Parallel, delayed

# function
def interpolate_meteo(root, ROI, met_var, ds):
    tinvData = xr.open_dataset(os.path.join(root, 'input', ROI + '_v2.nc'))
    date = pd.to_datetime(ds.time[0].values)
    filename = f"meteo_interp_{met_var}_{date.strftime('%Y%m%d_%p')}.nc"
    target = os.path.join(root, 'data', 'ERA5', ROI, filename)

    if not os.path.isfile(target):
        ds_interp = []
        for t_obs in ds.time:
            ds_t = ds.sel(time=t_obs)
            temp = ds_t.interp_like(tinvData)
            temp = temp.expand_dims('time')
            ds_interp.append(temp)

        ds_interp = xr.concat(ds_interp, dim='time')
        ds_interp.to_netcdf(target)
        print(f"Saved interpolated file: {target}")
    else:
        print(f"File already exists: {target}")


def main():
    root = input("Enter the root folder path: ").strip()
    if not os.path.isdir(root):
        print("Invalid root folder path. Exiting.")
        return

    ROI = input("Enter the Region of Interest (ROI1 or ROI2): ").strip()
    met_var = input("Enter the meteorological variable (t2m, v10, u10, cdir, tp, blh): ").strip()

    met_file = os.path.join(root, 'data', 'ERA5', f'met_{ROI}.nc')
    if not os.path.isfile(met_file):
        print(f"Meteorological data file not found: {met_file}")
        return

    mfdsMet = xr.open_dataset(met_file)

    Parallel(n_jobs=18)(
        delayed(interpolate_meteo)(root, ROI, met_var, mfdsMet[met_var].isel(time=slice(i, i + 12)))
        for i in np.arange(0, len(mfdsMet.time), 12)
    )


if __name__ == '__main__':
    import pandas as pd  
    main()
