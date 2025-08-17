import argparse
import os
import subprocess
import textwrap
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generator, Optional


class SentinelDataDownloader:
    BASE_URL = 'https://d1qb6yzwaaq4he.cloudfront.net/tropomi/no2'
    FILENAME_PATTERN = 'tropomi_no2_%Y%m%d.tar'
    DEFAULT_OUTPUT_DIR = './data/'
    
    def __init__(self, output_dir: str = DEFAULT_OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        
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
    
    def _construct_download_url(self, date: datetime) -> str:
        year = date.strftime('%Y')
        month = date.strftime('%m')
        filename = date.strftime(self.FILENAME_PATTERN)
        return f"{self.BASE_URL}/{year}/{month}/{filename}"
    
    def _get_target_path(self, date: datetime) -> Path:
        year = date.strftime('%Y')
        filename = date.strftime(self.FILENAME_PATTERN)
        return self.output_dir / 'Sentinelno2' / year / filename
    
    def _ensure_directory_exists(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
    
    def _execute_download(self, url: str, target_path: Path) -> bool:
        self._ensure_directory_exists(target_path.parent)
        
        wget_command = [
            'wget', '-q', '-r', '-l1', '-nd', '-nc', '--no-parent',
            f'--directory-prefix={target_path.parent}', url
        ]
        
        try:
            result = subprocess.run(wget_command, check=True, capture_output=True)
            return result.returncode == 0
        except subprocess.CalledProcessError:
            return False
    
    def _extract_archive(self, archive_path: Path) -> bool:
        if not archive_path.exists():
            return False
            
        tar_command = ['tar', '-xf', str(archive_path), '-C', str(archive_path.parent)]
        
        try:
            result = subprocess.run(tar_command, check=True, capture_output=True)
            return result.returncode == 0
        except subprocess.CalledProcessError:
            return False
    
    def _cleanup_archive(self, archive_path: Path) -> None:
        if archive_path.exists():
            try:
                archive_path.unlink()
            except OSError as e:
                print(f"Warning: Could not remove archive {archive_path}: {e}")
    
    def download_data_range(self, start_date: datetime, end_date: datetime) -> None:
        for date in self.generate_date_range(start_date, end_date):
            date_str = date.strftime('%Y-%m-%d')
            print(f"Downloading data for: {date_str}")
            
            url = self._construct_download_url(date)
            target_path = self._get_target_path(date)
            
            if self._execute_download(url, target_path):
                if self._extract_archive(target_path):
                    self._cleanup_archive(target_path)
                    print(f"Successfully processed: {date_str}")
                else:
                    print(f"Failed to extract archive for: {date_str}")
            else:
                print(f"Failed to download data for: {date_str}")


class ArgumentParser:
    @staticmethod
    def create_parser() -> argparse.ArgumentParser:
        description = textwrap.dedent("""
            Sentinel 5P TROPOMI NO2 Data Downloader
            
            Downloads Level 2 Sentinel 5P data from CloudFront distribution.
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
            help='Output directory path'
        )
        
        return parser


def validate_date_range(start_date: datetime, end_date: datetime) -> None:
    if start_date > end_date:
        raise ValueError("Start date must be before or equal to end date")


def main() -> int:
    try:
        parser = ArgumentParser.create_parser()
        args = parser.parse_args()
        
        start_date = SentinelDataDownloader.parse_date(args.starttime)
        end_date = SentinelDataDownloader.parse_date(args.stoptime)
        
        validate_date_range(start_date, end_date)
        
        downloader = SentinelDataDownloader(SentinelDataDownloader.DEFAULT_OUTPUT_DIR)
        downloader.download_data_range(start_date, end_date)
        
        return 0
        
    except ValueError as e:
        print(f"Date validation error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nDownload process interrupted by user")
        return 1
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        return 1


if __name__ == '__main__':
    exit(main())