# High-Resolution NO₂ Mapping and Modeling Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive machine learning pipeline for generating high-resolution nitrogen dioxide (NO₂) concentration maps by integrating satellite observations, meteorological data, emissions inventories, traffic patterns, land-use information, and topographic features.

## 🎯 Overview

This project provides a reproducible workflow for building hourly to monthly NO₂ concentration maps over Europe (or any region of interest) using TROPOMI/Sentinel-5P satellite data combined with multiple spatial predictors and machine learning techniques.

### Key Features

- **Multi-source data integration**: Combines satellite observations with meteorological, traffic, land-use, and topographic data
- **High spatial resolution**: 100m target grid resolution aligned with CORINE Land Cover
- **Temporal flexibility**: Supports hourly to monthly mapping
- **Machine learning pipeline**: Gradient-boosted trees (XGBoost) with advanced sampling strategies
- **Scalable processing**: Memory-efficient sub-region prediction and merging
- **Reproducible workflow**: Comprehensive data processing and modeling scripts

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Git
- Sufficient disk space for satellite and meteorological data (~100GB+ recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ml_project.git
   cd ml_project
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## 📁 Project Structure

```
ml_project/
├── README.md
├── requirements.txt
├── data/
│   └── traffic/
│       └── nuts3.xls
├── data_download/           # Data acquisition scripts
│   ├── download_airbase_stations.py
│   ├── download_era5_weather.py
│   ├── download_sentinel_no2.py
│   └── download_traffic_data.py
├── data_process/           # Data preprocessing pipeline
│   ├── feature_dem_topography.py
│   ├── feature_gaussian_convolution.py
│   ├── generate_master_dataset.py
│   ├── interpolate_meteorology.py
│   ├── interpolate_sentinel_hourly.py
│   ├── preprocess_airbase_stations.py
│   ├── preprocess_sentinel_reindex.py
│   └── rasterize_traffic_nuts3.py
├── input/                  # Reference data and metadata
│   ├── AIRBASE/
│   │   ├── Airbase_links.txt
│   │   └── metadata_AIRBASE.csv
│   └── LUD/
│       └── clc_legend.csv
├── model_train/            # Model training scripts
│   └── train_model.py
└── model_predict_map/      # Prediction and mapping
    ├── merge_predictions.py
    └── predict_maps.py
```

## 📊 Data Sources

### Primary Datasets

| Dataset | Source | Resolution | Purpose |
|---------|--------|------------|---------|
| **TROPOMI NO₂** | [TEMIS](https://www.temis.nl/airpollution/) | ~3.5×7 km | Satellite NO₂ observations |
| **ERA5 Meteorology** | [Copernicus CDS](https://cds.climate.copernicus.eu/) | ~30 km | Hourly meteorological reanalysis |
| **CORINE Land Cover** | [Copernicus Land](https://land.copernicus.eu/) | 100 m | Land use classification |
| **EU-DEM** | [EEA](https://www.eea.europa.eu/) | 25 m | Digital elevation model |
| **AirBase Stations** | [EEA](http://discomap.eea.europa.eu/) | Point data | Ground-truth NO₂ measurements |

### Auxiliary Datasets

- **Traffic Data**: OpenTransportMap road traffic intensity
- **Population Data**: Global Human Settlement (GHS-POP)
- **Emissions**: TNO/MACC-3 NOx point-source emissions

## 🔄 Processing Pipeline

### 1. Data Download
```bash
# Download satellite NO₂ data
python data_download/download_sentinel_no2.py

# Download meteorological data
python data_download/download_era5_weather.py

# Download air quality station data
python data_download/download_airbase_stations.py
```

### 2. Data Preprocessing
```bash
# Process and regrid satellite data to 100m resolution
python data_process/preprocess_sentinel_reindex.py 

# Interpolate satellite data to hourly resolution
python data_process/interpolate_sentinel_hourly.py 

# Extract topographic features using wavelet decomposition
python data_process/feature_dem_topography.py

# Interpolate meteorology to target grid
python data_process/interpolate_meteorology.py 
```

### 3. Feature Engineering
```bash
# Generate Gaussian convolution features for emissions
python data_process/feature_gaussian_convolution.py

# Rasterize traffic data to target grid
python data_process/rasterize_traffic_nuts3.py 

# Create master training dataset
python data_process/generate_master_dataset.py 
```

### 4. Model Training
```bash
# Train XGBoost model with stratified sampling
python model_train/train_model.py 
```

### 5. Prediction and Mapping
```bash
# Generate predictions for specific month and region
python model_predict_map/predict_maps.py 

# Merge sub-region predictions
python model_predict_map/merge_predictions.py 
```

## 🧠 Methodology

### Spatial Processing
- All predictors are regridded to a common 100m grid aligned with CORINE Land Cover
- Topographic feature extraction via 2D wavelet transforms
- Gaussian convolution applied to emissions data for distance-based features

### Machine Learning
- **Algorithm**: XGBoost gradient-boosted trees
- **Sampling Strategy**: Stratified sampling using UMAP clustering
- **Validation**: Spatial and temporal cross-validation
- **Features**: ~50+ predictors including meteorology, land use, traffic, topography, and emissions

## 🔧 System Requirements

### Minimum Requirements
- **RAM**: 16 GB
- **Storage**: 200 GB free space
- **CPU**: 4 cores
- **Python**: 3.8+

### Recommended Requirements
- **RAM**: 32 GB or more
- **Storage**: 500 GB+ SSD
- **CPU**: 8+ cores
- **GPU**: CUDA-compatible (for large-scale processing)

## 📋 Dependencies

### Core Scientific Libraries
```
numpy>=1.21.0
pandas>=1.3.0
xarray>=0.19.0
dask>=2021.6.0
scipy>=1.7.0
scikit-learn>=1.0.0
xgboost>=1.4.0
```

### Geospatial Libraries
```
cartopy>=0.20.0
shapely>=1.7.0
rasterio>=1.2.0
pyproj>=3.2.0
```

### Additional Libraries
```
matplotlib>=3.4.0
joblib>=1.0.0
requests>=2.25.0
PyWavelets>=1.1.0
umap-learn>=0.5.0
hdbscan>=0.8.0
```

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@software{no2_mapping_pipeline,
  title={High-Resolution NO₂ Mapping and Modeling Pipeline},
  author={Saket Kumar},
  year={2024},
  url={https://github.com/saket5ingh/DownNo2_XGBoost}
}
```

## 🙏 Acknowledgments

- European Space Agency (ESA) for Sentinel-5P TROPOMI data
- European Centre for Medium-Range Weather Forecasts (ECMWF) for ERA5 reanalysis
- European Environment Agency (EEA) for AirBase station data and land cover products
- The open-source scientific Python community

---

**Maintainers**: [Saket kumar](mailto:saketsingh9798@gmail.com)
**Last Updated**: August 2025
