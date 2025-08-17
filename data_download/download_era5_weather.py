#!/usr/bin/env python3

import argparse
import cdsapi
import os
import textwrap
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Generator, Union


class ERA5Downloader:
    DATASET_NAME = 'reanalysis-era5-single-levels'
    FILENAME_PATTERN = 'ERA5_ROI_%Y%m%d.nc'
    DEFAULT_OUTPUT_DIR = './data/'
    
    VARIABLES = [
        '10m_u_component_of_wind',
        '10m_v_component_of_wind',
        '2m_temperature',
        'clear_sky_direct_solar_radiation_at_surface',
        'total_precipitation'
    ]
    
    AREA_BOUNDS = [54, 2, 48, 8]
    
    HOURLY_TIMES = [
        '00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
        '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
        '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
        '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'
    ]
    
    def __init__(self, output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.client = None
        
    def _initialize_client(self) -> None:
        if self.client is None:
            self.client = cdsapi.Client()
    
    @staticmethod
    def parse_date(date_string: str) -> datetime:
        if len(date_string) == 10:
            return datetime.strptime(date_string, '%Y-%m-%d')
        return datetime.strptime(date_string, '%Y-%j')
    
    @staticmethod
    def generate_date_range(start_date: datetime, end_date: datetime) -> Generator[datetime, None, None]:
        current_date = start_date
        while current_date <= end_date:
            yield current_date
            current_date += timedelta(days=1)
    
    def _get_target_path(self, date: datetime) -> Path:
        filename = date.strftime(self.FILENAME_PATTERN)
        return self.output_dir / 'ERA5' / filename
    
    def _ensure_directory_exists(self, file_path: Path) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _build_request_parameters(self, date: datetime) -> Dict:
        return {
            'product_type': 'reanalysis',
            'variable': self.VARIABLES,
            'year': date.strftime('%Y'),
            'month': date.strftime('%m'),
            'day': date.strftime('%d'),
            'area': self.AREA_BOUNDS,
            'time': self.HOURLY_TIMES,
            'format': 'netcdf'
        }
    
    def _download_single_date(self, date: datetime) -> bool:
        target_path = self._get_target_path(date)
        
        if target_path.exists():
            print(f"File already exists: {target_path.name}")
            return True
        
        self._ensure_directory_exists(target_path)
        print(f"Downloading: {target_path.name}")
        
        try:
            self._initialize_client()
            request_params = self._build_request_parameters(date)
            self.client.retrieve(self.DATASET_NAME, request_params, str(target_path))
            print(f"Successfully downloaded: {target_path.name}")
            return True
        except Exception as e:
            print(f"Failed to download {target_path.name}: {e}")
            return False
    
    def download_date_range(self, start_date: datetime, end_date: datetime) -> None:
        successful_downloads = 0
        total_dates = 0
        
        for date in self.generate_date_range(start_date, end_date):
            total_dates += 1
            if self._download_single_date(date):
                successful_downloads += 1
        
        print(f"\nDownload summary: {successful_downloads}/{total_dates} files processed successfully")


class ArgumentParser:
    @staticmethod
    def create_parser() -> argparse.ArgumentParser:
        description = textwrap.dedent("""
            ERA5 Reanalysis Data Downloader
            
            Downloads ERA5 single-level reanalysis data from Copernicus Climate Data Store.
            Includes: wind components, temperature, solar radiation, and precipitation.
            Supports date input in YYYY-mm-dd or YYYY-jjj format.
        """)
        
        parser = argparse.ArgumentParser(
            description=description,
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        parser.add_argument(
            'starttime',
            type=str,
            help='Start date (YYYY-mm-dd or YYYY-jjj)'
        )
        
        parser.add_argument(
            'stoptime',
            type=str,
            help='Stop date (YYYY-mm-dd or YYYY-jjj)'
        )
        
        parser.add_argument(
            '--prefix',
            default='.',
            type=str,
            help='Output directory prefix'
        )
        
        return parser


def validate_date_range(start_date: datetime, end_date: datetime) -> None:
    if start_date > end_date:
        raise ValueError("Start date must be before or equal to end date")


def main() -> int:
    try:
        parser = ArgumentParser.create_parser()
        args = parser.parse_args()
        
        start_date = ERA5Downloader.parse_date(args.starttime)
        end_date = ERA5Downloader.parse_date(args.stoptime)
        
        validate_date_range(start_date, end_date)
        
        downloader = ERA5Downloader(ERA5Downloader.DEFAULT_OUTPUT_DIR)
        downloader.download_date_range(start_date, end_date)
        
        return 0
        
    except ValueError as e:
        print(f"Date validation error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nDownload process interrupted by user")
        return 1
    except ImportError:
        print("Error: cdsapi module not found. Install with: pip install cdsapi")
        return 1
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        return 1


if __name__ == '__main__':
    exit(main())