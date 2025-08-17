import os
import glob
import pandas as pd
import shapefile
import numpy as np
import xarray as xr
from shapely.geometry import MultiLineString, Polygon, box
from joblib import Parallel, delayed


def make_grid_cells(pos):
    polygon = [
        (pos[0]-pos[2], pos[1]-pos[3]),
        (pos[0]+pos[2], pos[1]-pos[3]),
        (pos[0]+pos[2], pos[1]+pos[3]),
        (pos[0]-pos[2], pos[1]+pos[3]),
        (pos[0]-pos[2], pos[1]-pos[3])
    ]
    return Polygon(polygon)


def extract_traffic_info(root, ROI, ROIbox, thres, nuts_id):
    target = os.path.join(root, ROI, f'roadlinks_{nuts_id}.nc')
    if os.path.isfile(target):
        return None

    shp_path = os.path.join(root, 'traffic', nuts_id, f'roadlinks_{nuts_id}.shp')
    dbf_path = os.path.join(root, 'traffic', nuts_id, f'roadlinks_{nuts_id}.dbf')

    if not os.path.isfile(shp_path) or not os.path.isfile(dbf_path):
        print(f"Shapefile or DBF missing for {nuts_id}")
        return None

    with open(shp_path, 'rb') as myshp, open(dbf_path, 'rb') as mydbf:
        r = shapefile.Reader(shp=myshp, dbf=mydbf, encoding='latin-1')
        rbox = box(r.bbox[0], r.bbox[1], r.bbox[2], r.bbox[3])

        if not rbox.intersects(ROIbox):
            return None

        try:
            spatial_coords = xr.open_dataset(os.path.join(root, 'input', 'OSMtraffic', f'spatial_coords_{ROI}.nc'))
            nut3region = spatial_coords.sel(
                lon=slice(r.bbox[0], r.bbox[2]),
                lat=slice(r.bbox[1], r.bbox[3])
            )

            roadlist = r.shapes()
            roadinfo = r.records()

            for info, roadname in zip(roadinfo, roadlist):
                roadlines = MultiLineString([roadname.points])
                roadgrids = nut3region.sel(
                    lon=slice(roadlines.bounds[0]-thres, roadlines.bounds[2]+thres),
                    lat=slice(roadlines.bounds[1]-thres, roadlines.bounds[3]+thres)
                )

                grid_cells = np.stack([
                    roadgrids.lonvV.values.ravel(),
                    roadgrids.latvV.values.ravel(),
                    roadgrids.lonvD.values.ravel(),
                    roadgrids.latvD.values.ravel()
                ], axis=1)

                totLength = roadlines.length
                tvol = info.get('trafficvol', -1)

                for pos in grid_cells:
                    grid_cell = make_grid_cells(pos)
                    intersectlength = roadlines.intersection(grid_cell).length
                    nut3region.linelength.loc[pos[1], pos[0]] += intersectlength
                    try:
                        nut3region.trafficvolume.loc[pos[1], pos[0]] += tvol * intersectlength / totLength
                    except Exception:
                        pass

            nut3region = nut3region.chunk(chunks={'lon': 256, 'lat': 256})
            nut3region.to_netcdf(target)

        except Exception as e:
            print(f"Failed processing {nuts_id}: {e}")

    return None


def main():
    root = input("Enter the root folder path: ").strip()
    if not os.path.isdir(root):
        print("Invalid root folder path. Exiting.")
        return

    ROI = input("Enter the Region of Interest (ROI1 or ROI2): ").strip()

    
    dataSpatial = xr.open_dataset(os.path.join(root, 'input', f'{ROI}.nc'))
    lat = dataSpatial.lat
    lon = dataSpatial.lon
    devlon = (lon - np.roll(lon, 1)) * 0.5
    devlon[0] = devlon[1]
    devlat = (lat - np.roll(lat, 1)) * 0.5
    devlat[0] = devlat[1]
    thres = 0.001
    ROIbox = box(lon[0]-devlon[0], lat[0]-devlat[0], lon[-1]+devlon[-1], lat[-1]+devlat[-1])

    
    df = pd.read_excel(os.path.join(root, 'input', 'OSMtraffic', 'nuts3.xls'))
    ids = df['NUTS 3 ID (2010)'].dropna().values

    
    Parallel(n_jobs=-1)(delayed(extract_traffic_info)(root, ROI, ROIbox, thres, idn) for idn in ids)

    
    filename = os.path.join(root, 'input', 'OSMtraffic', f'spatial_coords_{ROI}.nc')
    raster_traffic = xr.open_dataset(filename, drop_variables={'lonvV', 'latvV', 'lonvD', 'latvD'})
    raster_traffic.linelength[:] = np.nan
    raster_traffic.trafficvolume[:] = np.nan
    length = np.full_like(raster_traffic.linelength, np.nan)
    volume = np.full_like(raster_traffic.linelength, np.nan)

    pathname = os.path.join(root, ROI, 'roadlinks_*.nc')
    filenames = glob.glob(pathname)

    for f in filenames:
        ds = xr.open_dataset(f, drop_variables={'lonvV', 'latvV', 'lonvD', 'latvD'})
        _, dsa = xr.align(raster_traffic, ds, join='left')
        bdsa = (dsa.linelength.values > 0)
        length[bdsa] = dsa.linelength.values[bdsa]
        volume[bdsa] = dsa.trafficvolume.values[bdsa]
        print(f)

    raster_traffic.linelength[:] = length
    raster_traffic.trafficvolume[:] = volume

    target = os.path.join(root, ROI, 'traffic_raster.nc')
    raster_traffic.to_netcdf(target)
    print(f"Saved combined raster: {target}")


if __name__ == "__main__":
    main()
