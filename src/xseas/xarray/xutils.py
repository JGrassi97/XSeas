"""
Utility functions for xarray operations in seasonal analysis.

This module provides functions for generating seasonal labels and predictions
from breakpoint data using xarray's apply_ufunc capabilities.
"""
from typing import Dict, Any
import numpy as np
import xarray as xr


def get_prediction(breakpoints: np.ndarray, **kwargs) -> np.ndarray:
    """
    Generate seasonal predictions from breakpoints.
    
    Parameters
    ----------
    breakpoints : np.ndarray
        Array of breakpoint values defining season boundaries.
    **kwargs : dict
        Additional parameters including:
        - n_seas : int, number of seasons
        
    Returns
    -------
    np.ndarray
        Array of seasonal labels for each day of year (365 days).
        Returns NaN array if computation fails.
    """
    n_seas = kwargs.get('n_seas', 2)
    
    if np.any(np.isnan(breakpoints)):
        return np.full(365, np.nan)
    
    try:
        prediction = np.zeros(365, dtype=int)
        indices = _generate_season_indices(breakpoints, n_seas)
        
        for season_id, idx in enumerate(indices):
            prediction[idx.astype(int)] = season_id
            
        return prediction
    
    except Exception:
        return np.full(365, np.nan)


def _generate_season_indices(breakpoints: np.ndarray, n_seas: int) -> list:
    """
    Generate day-of-year indices for each season.
    
    Parameters
    ----------
    breakpoints : np.ndarray
        Breakpoint values defining season boundaries.
    n_seas : int
        Number of seasons.
        
    Returns
    -------
    list
        List of numpy arrays, each containing day indices for a season.
    """
    if n_seas == 1:
        return [np.arange(365)]
    
    indices = []
    sorted_bp = np.sort(breakpoints)
    
    for i in range(n_seas):
        start = int(sorted_bp[i])
        end = int(sorted_bp[(i + 1) % n_seas])
        
        if start >= end:  # Wraps around year
            season_idx = np.concatenate([
                np.arange(start, 365),
                np.arange(0, end)
            ])
        else:
            season_idx = np.arange(start, end)
            
        indices.append(season_idx)
    
    return indices


def generate_labels(breakpoints: xr.DataArray, **kwargs) -> xr.DataArray:
    """
    Apply seasonal labeling to breakpoint data using xarray.apply_ufunc.
    
    Parameters
    ----------
    breakpoints : xr.DataArray
        DataArray containing breakpoint values with 'cluster' dimension.
    **kwargs : dict
        Parameters passed to get_prediction function.
        
    Returns
    -------
    xr.DataArray
        DataArray with seasonal labels for each day of year.
    """
    labels = xr.apply_ufunc(
        get_prediction,
        breakpoints,
        kwargs=kwargs,
        vectorize=True,
        dask="parallelized",
        input_core_dims=[["cluster"]],
        output_core_dims=[["dayofyear"]],
        dask_gufunc_kwargs={"output_sizes": {"dayofyear": 365}},
        output_dtypes=[int],
        keep_attrs=True
    )
    return labels


def tile_labels(dataset: xr.Dataset, labels: xr.Dataset, n_seasons: int) -> xr.Dataset:
    """
    Tile seasonal labels across all years in a dataset.
    
    Parameters
    ----------
    dataset : xr.Dataset
        Input dataset with time dimension.
    labels : xr.Dataset
        Dataset containing breakpoint information.
    n_seasons : int
        Number of seasons.
        
    Returns
    -------
    xr.Dataset
        Dataset with added 'labels' variable containing seasonal classifications.
    """
    label_params = {'n_seas': n_seasons}
    seasonal_labels = generate_labels(labels['breakpoints'], **label_params)
    
    # Get time dimension info from the dataset
    time_dim = dataset.time
    n_time_steps = len(time_dim)
    
    # Calculate number of complete years (assuming 365 days per year)
    days_per_year = 365
    n_years = n_time_steps // days_per_year
    
    if n_time_steps % days_per_year != 0:
        print(f"Warning: Dataset has {n_time_steps} time steps, which is not a multiple of {days_per_year} days")
    
    # Get the spatial dimensions
    lat_dim = dataset.lat
    lon_dim = dataset.lon
    
    # Tile labels across years and reshape to match dataset dimensions
    label_values = seasonal_labels.values  # Shape: (lat, lon, dayofyear=365)
    
    # Repeat for each year and reshape to (lat, lon, time)
    tiled_labels = np.tile(label_values, (1, 1, n_years))
    
    # Trim to exact time dimension if necessary
    if n_time_steps < tiled_labels.shape[2]:
        tiled_labels = tiled_labels[:, :, :n_time_steps]
    elif n_time_steps > tiled_labels.shape[2]:
        # Pad with last year's pattern if needed
        remaining_days = n_time_steps - tiled_labels.shape[2]
        padding = label_values[:, :, :remaining_days]
        tiled_labels = np.concatenate([tiled_labels, padding], axis=2)
    
    # Add to dataset with correct dimensions
    dataset = dataset.copy()
    dataset['labels'] = (('lat', 'lon', 'time'), tiled_labels)
    
    return dataset


# Backward compatibility
X_labels = generate_labels  # Alias for backward compatibility