# ml_project

High-resolution NO₂ mapping and modeling pipeline combining satellite observations, meteorology, emissions, traffic, land-use and topography with machine learning. This repository contains scripts for data download, data processing, model training, and map generation.

> **Why this exists**: to replicate a reproducible workflow for building hourly to monthly NO₂ concentration maps over Europe (or your region of interest) using TROPOMI/Sentinel-5P and a stack of spatial predictors.



## Quick start

```bash
# 1) Create env
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# 2) Install requirements
pip install --upgrade pip
pip install -r requirements.txt

# 3) Configure credentials (optional; only for ERA5 etc.)
# export CDSAPI_URL=...
# export CDSAPI_KEY=...
# export MAPBOX_TOKEN=...      # if you plot maps with mapbox (optional)
```


## Project structure

```
ml_project/
└── ml_project
    ├── data
    │   └── traffic
    │       └── nuts3.xls
    ├── data_download
    │   ├── download_airbase_stations.py
    │   ├── download_era5_weather.py
    │   ├── download_sentinel_no2.py
    │   └── download_traffic_data.py
    ├── data_process
    │   ├── feature_dem_topography.py
    │   ├── feature_gaussian_convolution.py
    │   ├── generate_master_dataset.py
    │   ├── interpolate_meteorology.py
    │   ├── interpolate_sentinel_hourly.py
    │   ├── preprocess_airbase_stations.py
    │   ├── preprocess_sentinel_reindex.py
    │   └── rasterize_traffic_nuts3.py
    ├── input
    │   ├── AIRBASE
    │   │   ├── Airbase_links.txt
    │   │   └── metadata_AIRBASE.csv
    │   └── LUD
    │       └── clc_legend.csv
    ├── model_predict_map
    │   ├── merge_predictions.py
    │   └── predict_maps.py
    ├── model_train
    │   └── train_model.py
    ├── README.md
    └── requirements.txt
```


## Data download

This work includes multiple datasets:

- **TROPOMI NO₂ (Sentinel-5P L2)** – satellite observations of NO₂.  
  Source: https://www.temis.nl/airpollution/  
  Script: `./data_download/sentinel_data_down.py` (if present).

- **Meteorology (ERA5, hourly single levels)** – reanalysis.  
  Source: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview  
  Script: `./data_download/ERA5_data_down.py` (if present).

- **Traffic (OpenTransportMap)** – road traffic intensity over Europe.  
  Source: http://opentransportmap.info/  
  Script: `./data_download/opentf_data_down.py` (if present).

- **AirBase (EEA AQ stations)** – near-surface NO₂ for training/validation.  
  Source: http://discomap.eea.europa.eu/map/fme/AirQualityExport.htm  
  Script: `./data_download/airbase_data_down.py` (if present).  
  Uses pre-saved links in `./input/AIRBASE/Airbase_links.txt` (if present).

- **Land use (CORINE Land Cover 2018, 100 m)**  
  Source: https://land.copernicus.eu/en/products/corine-land-cover/clc2018

- **Topography (EU-DEM v1.1)**  
  Source: https://www.eea.europa.eu/en/datahub/datahubitem-view/d08852bc-7b5f-4835-a776-08362e2fbf4b

- **Population (GHS / JRC)**  
  Source: https://publications.jrc.ec.europa.eu/repository/handle/JRC100523  
  (Other global population datasets at finer resolution can also be used.)

- **NOx point-source emissions (TNO/MACC-3)**  
  Reference: https://acp.copernicus.org/articles/14/10963/2014/


## Data processing

All predictors are regridded/interpolated to a **100 m target grid** aligned with CORINE Land Cover. Key steps:

- **Gap-filling & regridding of Sentinel-5P** using methods inspired by Kuhlmann et al. (2014): https://amt.copernicus.org/articles/7/451/2014/  
  - Script: `./data_process/Sentinel_down_gridding.py`
  - Alternative simple regridding with Xarray reindexing: `./data_process/Sentinel_reindexing.py`
  - Hourly linear interpolation from daily: `./data_process/sentinel_hourly_interpolation.py`

- **Length-wise decomposition of topographic features** via 2D wavelet transforms: `./data_process/dem_features.py`

- **Spatial interpolation of ERA5** from ~30 km to 100 m: `./data_process/meteo_interp.py`

- **Rasterisation of traffic vectors**: `./data_process/nut2_traffic_rasterise.py`

- **Gaussian convolution of emissions** to separate magnitude & distance effects: `./data_process/data_gaussian_convolution.py`

- **AQ station cleaning & aggregation**: `./data_process/airbase_data_processing.py`

- **Core dataset assembly** for model training: `./data_process/generate_core_data.py`


## Modeling

- **Train**: gradient-boosted trees (XGBoost) with stratified sampling using UMAP clustering.  
  Script: `./model_train/xgb_train_save.py`

- **Predict & map**:  
  1. Predict sub-regions to avoid memory pressure: `./model_predict_map/predict_maps_month.py`  
  2. Merge sub-regions to a single NetCDF and plot: `./model_predict_map/merge_predicted_ROI_maps.py`


## Usage examples

```bash
# Example: process Sentinel-5P to hourly fields
python data_process/sentinel_hourly_interpolation.py --input path/to/daily.nc --output path/to/hourly.nc

# Example: train model
python model_train/xgb_train_save.py --config configs/xgb.yaml

# Example: produce monthly map tiles for a ROI
python model_predict_map/predict_maps_month.py --month 2020-06 --roi configs/roi_europe.geojson
```


## Credentials & environment

Some downloads require accounts/tokens:

- **ERA5 (CDS API)**: create `~/.cdsapirc` or set `CDSAPI_URL` and `CDSAPI_KEY`.
- **Optional** map visualisation tokens (e.g., `MAPBOX_TOKEN`).

You can configure secrets via environment variables or a `.env` file (if `python-dotenv` is installed).


## Requirements

Install from the generated `requirements.txt` (auto-derived by statically scanning imports in the repo).  
If something is missing for your platform (e.g., `cartopy`, `rasterio`), please install system libraries as required by those packages.

**Detected third‑party imports (17):**
```
cartopy, cdsapi, dask, hdbscan, joblib, matplotlib, numpy, pandas, pywt, requests, scipy, shapefile, shapely, sklearn, umap, xarray, xgboost
```

**Proposed pip requirements (17):**
```
PyWavelets
cartopy
cdsapi
dask
hdbscan
joblib
matplotlib
numpy
pandas
requests
scikit-learn
scipy
shapefile
shapely
umap-learn
xarray
xgboost
```


## Reproducibility & tips

- Prefer pinned versions (e.g., `package==x.y.z`) for long-term reproducibility once your environment is working.
- Use `mamba` or `conda` for heavy geospatial stacks if wheels are not available for your OS.
- Keep large raw datasets outside of the repo; point scripts to data folders via configs.

## License

Add your preferred license (e.g., MIT) here.
