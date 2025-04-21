# ERA5 Evening Temperature Change US

This repository contains code to analyze temperature changes in the United States during evening hours (8 PM to midnight local time) using ERA5 hourly 2m temperature data.

## Overview

The analysis calculates temperature changes from a 1980-1989 baseline for each 5-year period up to 2024. It focuses specifically on evening hours (8 PM to midnight) in local time at each location, using a precise time zone mapping for the continental United States.

## Features

- Processes ERA5 hourly 2m temperature data
- Uses precise time zone mapping for the continental US
- Extracts evening hours (8 PM to midnight) in local time
- Calculates temperature changes from a 1980-1989 baseline
- Aggregates results by US state
- Exports results to Excel

## Requirements

- Python 3.6+
- Required packages:
  - numpy
  - pandas
  - xarray (may require additional dependencies like netCDF4 or h5netcdf for working with NetCDF files)
  - geopandas
  - matplotlib
  - openpyxl

## Data Requirements

The script expects ERA5 hourly 2m temperature data files in NetCDF format in the `data` directory. Files should be named according to the pattern:

```
era5_hourly_2m_temperature_YYYY.nc
```

where `YYYY` is the year (e.g., `era5_hourly_2m_temperature_1980.nc`).

The script also requires state boundary shapefiles in the `data/ne_10m_admin_1_states_provinces` directory.

## Usage

1. Place ERA5 data files in the `data` directory
2. Run the script:

```bash
python process_era5_temp.py
```

3. Results will be saved in the `output` directory, including:
   - Hourly averages for the baseline and each period
   - Temperature changes for each period
   - An Excel file with temperature changes by state

## Time Zone Mapping

The script uses a precise time zone mapping for the continental United States:
- Pacific Time (UTC-8): -125° to -115° longitude
- Mountain Time (UTC-7): -115° to -101° longitude
- Central Time (UTC-6): -101° to -87° longitude
- Eastern Time (UTC-5): -87° to -67° longitude

For locations outside these defined boundaries, it falls back to the longitude/15 calculation.
