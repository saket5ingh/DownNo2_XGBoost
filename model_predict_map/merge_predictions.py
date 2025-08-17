

import os
import xarray as xr
import glob
import numpy as np
import argparse
import textwrap
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.colors import LightSource
from datetime import datetime, timedelta
from joblib import Parallel, delayed


def parse_date(date_string):
    """Parse date string in YYYY-MM-DD or YYYY-DDD format."""
    if len(date_string) == 10:
        return datetime.strptime(date_string, '%Y-%m-%d')
    else:
        return datetime.strptime(date_string, '%Y-%j')


def iter_dates(start_date, end_date):
    """Generate dates between start and end dates."""
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def create_colormap():
    """Create custom colormap for NO2 visualization."""
    cm0 = LinearSegmentedColormap.from_list(
        'cm0', ['midnightblue', 'darkturquoise', (1, 1, 0.5)], N=64
    )
    cm1 = LinearSegmentedColormap.from_list(
        'cm1', [(1, 1, 0.5), 'darkorange', 'firebrick', 'maroon'], N=192
    )
    
    newcolors = np.vstack((
        cm0(np.linspace(0, 1, 128)),
        cm1(np.linspace(0, 1, 128))
    ))
    return ListedColormap(newcolors)


def create_hillshade(data_path):
    """Create hillshade for topographic background."""
    elevation_file = os.path.join(data_path, 'input', 'ROI1_v2.nc')
    dataset = xr.open_dataset(elevation_file)
    elevation = dataset.dem.data
    
    light_source = LightSource(azdeg=315, altdeg=45)
    vertical_exaggeration = 0.3
    
    return light_source.hillshade(
        elevation, 
        vert_exag=vertical_exaggeration, 
        dx=20, 
        dy=20
    )


def plot_and_save_figure(dataset, filename, colormap, hillshade, output_dir):
    """Generate and save NO2 concentration map figure."""
    print(f"Creating figure: {filename}")
    
    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), 
                   alpha=0.4, edgecolor='k', zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), 
                   alpha=0.4, edgecolor='k', zorder=3)
    
    no2_plot = ax.imshow(
        dataset.no2[0, :, :], 
        cmap=colormap, 
        aspect=1, 
        vmin=0, 
        vmax=60, 
        extent=[6, 12, 42, 48], 
        zorder=1, 
        origin='lower'
    )
    
    ax.imshow(
        hillshade, 
        aspect=1, 
        extent=[6, 12, 42, 48], 
        cmap='gray', 
        alpha=0.1, 
        zorder=2, 
        origin='lower', 
        interpolation='None'
    )
    
    ax.set_xticks(np.arange(6, 13, 1))
    ax.set_yticks(np.arange(42, 49, 1))
    ax.set_xlim(6, 12)
    ax.set_ylim(44, 48)
    
    colorbar = fig.colorbar(
        no2_plot, 
        ax=ax, 
        extend='max', 
        shrink=0.75, 
        pad=0.03, 
        label='Near-surface NO₂ concentration [µg·m⁻³]'
    )
    
    time_string = str(dataset.time[0].values).replace('T', ' ')[:19]
    ax.set_title(time_string, fontsize=14)
    ax.set_xlabel('Longitude (°E)', fontsize=12)
    ax.set_ylabel('Latitude (°N)', fontsize=12)
    
    colorbar.set_alpha(1)
    colorbar.draw_all()
    
    output_path = os.path.join(output_dir, f'{filename}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close('all')


def combine_and_visualize_maps(date, data_path, model_id, model_name, colormap, hillshade):
    """Combine prediction maps for all hours of a given date and create visualizations."""
    roi = 'ROI1'
    
    for hour in range(24):
        file_pattern = f'{roi}_{model_name}_no2_{date.strftime("%Y-%m-%d")}T{hour:02d}_*.nc'
        base_filename = file_pattern.split('*')[0][:-1]
        merged_filename = f'{base_filename}.nc'
        
        hourly_dir = os.path.join(data_path, 'models', model_id, 'Results', 'Hourly')
        figures_dir = os.path.join(hourly_dir, 'Figures')
        
        os.makedirs(hourly_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)
        
        merged_file_path = os.path.join(hourly_dir, merged_filename)
        
        if not os.path.isfile(merged_file_path):
            results_dir = os.path.join(data_path, 'models', model_id, 'Results')
            file_list = glob.glob(os.path.join(results_dir, file_pattern))
            
            if len(file_list) == 36:
                try:
                    with xr.open_mfdataset(file_list, combine='by_coords') as dataset:
                        dataset.to_netcdf(merged_file_path)
                        plot_and_save_figure(
                            dataset, base_filename, colormap, hillshade, figures_dir
                        )
                        print(f"Successfully processed: {base_filename}")
                except Exception as e:
                    print(f'Error processing file {base_filename}: {str(e)}')
            else:
                print(f'Missing data for {base_filename}: found {len(file_list)}/36 files')
        else:
            print(f'File already exists: {base_filename}')


def validate_inputs(data_path, model_name, qa_threshold):
    """Validate input parameters and paths."""
    if not os.path.exists(data_path):
        raise ValueError(f"Data path does not exist: {data_path}")
    
    valid_models = ['Model0', 'Model1', 'Model2', 'Model3', 'Model4']
    if model_name not in valid_models:
        raise ValueError(f"Invalid model name. Must be one of: {valid_models}")
    
    if qa_threshold not in [50, 75]:
        raise ValueError("QA threshold must be 50 or 75")
    
    input_dir = os.path.join(data_path, 'input')
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    elevation_file = os.path.join(input_dir, 'ROI1_v2.nc')
    if not os.path.exists(elevation_file):
        raise ValueError(f"Elevation data file not found: {elevation_file}")


def main():
    description = textwrap.dedent("""\
        Merge predicted NO2 concentration maps and generate visualization figures.
        
        This script combines sub-regional prediction maps (1°x1° grid cells) into 
        complete hourly maps for the region of interest and creates publication-ready 
        figures with topographic background.
        
        Available Models:
        - Model0: Default configuration
        - Model1: Null model without feature extraction
        - Model2: Spatial features with emission source extraction  
        - Model3: Spatial features + wavelet transform of DEM
        - Model4: Dimensionality reduction (selected features)
        
        Output:
        - Merged NetCDF files in: data_path/models/model_id/Results/Hourly/
        - Figure files in: data_path/models/model_id/Results/Hourly/Figures/
    """)
    
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('data_path', type=str, 
                       help='Path to the data directory')
    parser.add_argument('model_name', type=str,
                       choices=['Model0', 'Model1', 'Model2', 'Model3', 'Model4'],
                       help='Model configuration name')
    parser.add_argument('start_date', type=str,
                       help='Start date (YYYY-MM-DD or YYYY-DDD format)')
    parser.add_argument('end_date', type=str,
                       help='End date (YYYY-MM-DD or YYYY-DDD format)')
    parser.add_argument('qa_threshold', type=int, choices=[50, 75],
                       help='Quality assurance threshold (50 or 75)')
    parser.add_argument('--n_jobs', type=int, default=-1,
                       help='Number of parallel jobs (-1 for all cores)')
    
    args = parser.parse_args()
    
    try:
        validate_inputs(args.data_path, args.model_name, args.qa_threshold)
        
        start_date = parse_date(args.start_date)
        end_date = parse_date(args.end_date)
        
        if start_date > end_date:
            raise ValueError("Start date must be before or equal to end date")
        
        model_id = f'Full_features{args.qa_threshold}'
        colormap = create_colormap()
        hillshade = create_hillshade(args.data_path)
        
        print(f"Processing dates from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"Using model: {args.model_name}")
        print(f"QA threshold: {args.qa_threshold}")
        print(f"Data path: {args.data_path}")
        
        date_list = list(iter_dates(start_date, end_date))
        print(f"Total dates to process: {len(date_list)}")
        
        Parallel(n_jobs=args.n_jobs)(
            delayed(combine_and_visualize_maps)(
                date, args.data_path, model_id, args.model_name, colormap, hillshade
            ) for date in date_list
        )
        
        print("Map merging and visualization completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())