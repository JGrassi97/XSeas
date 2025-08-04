import numpy as np
import xarray as xr
import warnings
from tqdm import tqdm
import argparse
import os
import yaml

from xseas.manager.utils import load_variables

# Suppress warnings
warnings.filterwarnings("ignore")


def rolling_doy_complete(da, window_size=30):
    """Apply rolling window averaging by day of year."""
    doy = da.time.dt.dayofyear  
    zscore_da = xr.full_like(da, np.nan) 
    
    for d in tqdm(range(1, 366)):  
        doy_mask = doy == d
        times_doy = da.time[doy_mask]
        
        if len(times_doy) < 2:
            continue

        for i, t in enumerate(times_doy):
            start = max(0, i - window_size // 2)
            end = min(len(times_doy), i + window_size // 2 + 1)  
            window_doy = da.sel(time=times_doy[start:end])
            mean_doy = window_doy.mean(dim='time')
            zscore_da.loc[dict(time=t)] = (mean_doy)

    return zscore_da


def rolling_zscore_complete(da, window_size=30):
    """Apply rolling z-score normalization."""
    times = da.time.values
    zscore_da = xr.full_like(da, np.nan)
    
    for i, t in tqdm(enumerate(times)):
        start = max(0, i - window_size // 2)
        end = min(len(times), i + window_size // 2)
        window = da.isel(time=slice(start, end))
        
        mean = window.mean(dim='time')
        std = window.std(dim='time')

        try:
            zscore_da.loc[dict(time=t)] = (da.sel(time=t) - mean) / std
        except:
            zscore_da.loc[dict(time=t)] = 0

    return zscore_da


def normalize_ERA5(base_path, variables, variables_codes):
    """
    Normalize ERA5 data using rolling window techniques.
    
    Parameters
    ----------
    base_path : str
        Path to ERA5 data directory.
    variables : List[str]
        List of variable names (not used for ERA5, but kept for consistency).
    variables_codes : List[str]
        List of ERA5 variable codes (e.g., ['2t', 'tp', 'u', 'v']).
        
    Returns
    -------
    xarray.Dataset
        Normalized ERA5 dataset.
    """
    print("🔄 Loading ERA5 datasets...")
    
    datasets = []
    
    # Load each variable
    for var_code in variables_codes:
        file_path = os.path.join(base_path, f'{var_code}.nc')
        
        if not os.path.exists(file_path):
            print(f"⚠️  Warning: ERA5 file not found: {file_path}")
            continue
            
        try:
            print(f"📂 Loading {var_code} from {file_path}")
            
            # Load dataset
            ds = xr.open_dataset(file_path)
            
            # Get the main variable (handle different naming conventions)
            if var_code in ds.variables:
                da = ds[var_code]
            elif var_code == '2t' and 't2m' in ds.variables:
                da = ds['t2m']
            elif var_code == 'tp' and 'total_precipitation' in ds.variables:
                da = ds['total_precipitation']
            else:
                # Try to find the main data variable
                data_vars = [v for v in ds.data_vars if len(ds[v].dims) >= 3]
                if data_vars:
                    da = ds[data_vars[0]]
                    print(f"   Using variable '{data_vars[0]}' for {var_code}")
                else:
                    print(f"❌ Could not find suitable variable for {var_code}")
                    continue
            
            # Ensure we have time, lat, lon dimensions
            required_dims = ['time', 'lat', 'lon']
            if not all(dim in da.dims for dim in required_dims):
                print(f"❌ Missing required dimensions for {var_code}: {da.dims}")
                continue
            
            # Rename to standard name for consistency
            da.name = var_code
            datasets.append(da)
            print(f"✅ Loaded {var_code}: {da.shape}")
            
        except Exception as e:
            print(f"❌ Error loading {var_code}: {e}")
            continue
    
    if not datasets:
        raise ValueError("No ERA5 datasets could be loaded")
    
    # Merge all datasets
    print("🔄 Merging datasets...")
    dataset_merged = xr.merge(datasets).load()
    print(f"✅ Merged dataset shape: {dataset_merged.dims}")
    
    # Apply normalization steps
    print("🔄 Applying day-of-year rolling average (window=15 days)...")
    dataset_doy = rolling_doy_complete(dataset_merged, window_size=15)
    
    print("🔄 Applying rolling z-score normalization (window=10 years)...")
    dataset_normalized = rolling_zscore_complete(dataset_doy, window_size=365*10)
    
    # Add metadata
    dataset_normalized.attrs.update({
        'description': 'ERA5 data normalized with rolling window techniques',
        'normalization_steps': [
            'Day-of-year rolling average (15 days)',
            'Rolling z-score normalization (10 years)'
        ],
        'created_by': 'XSeas normalize_ERA5',
        'variables': list(variables_codes)
    })
    
    print("✅ ERA5 normalization completed")
    return dataset_normalized


def normalize_CMIP6(base_path, variables, variables_codes, scenarios):
    """
    Normalize CMIP6 data using rolling window techniques.
    
    Parameters
    ----------
    base_path : str
        Path to CMIP6 model directory.
    variables : List[str]
        List of variable directory names.
    variables_codes : List[str]
        List of variable codes in NetCDF files.
    scenarios : List[str]
        List of scenarios to process.
        
    Returns
    -------
    xarray.Dataset
        Normalized CMIP6 dataset.
    """
    normalized_dataset = []

    for scenario in scenarios:
        dataset = load_variables(os.path.join(base_path, scenario), variables, variables_codes)
        for dat in dataset:
            normalized_dataset.append(dat)

    dataset_proj = xr.merge(normalized_dataset).load()
    dataset_proj = rolling_doy_complete(dataset_proj, window_size=15)
    dataset_proj = rolling_zscore_complete(dataset_proj, window_size=365*10)
    
    return dataset_proj