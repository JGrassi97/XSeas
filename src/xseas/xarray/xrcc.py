"""
Xarray interface for Radially Constrained Clustering (RCC).

This module provides xarray integration for the RCC algorithm, enabling
distributed computation across spatial grids using dask.
"""
from typing import Tuple, List, Optional, Union
import numpy as np
import xarray as xr
from sklearn.metrics import silhouette_score

from xseas.models import RCC


def _cluster_gridpoint(*grid_points: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply RCC clustering to individual grid points.
    
    Parameters
    ----------
    *grid_points : np.ndarray
        Variable arrays for a single grid point.
    **kwargs : dict
        Clustering parameters.
        
    Returns
    -------
    Tuple of arrays containing breakpoints, error history, and silhouette scores.
    """
    # Extract parameters
    n_iter = kwargs.get('iters', 20)
    n_seas = kwargs.get('n_seas', 2)
    learning_rate = kwargs.get('learning_rate', 10)
    min_len = kwargs.get('min_len', 30)
    starting_bp = kwargs.get('starting_bp', None)
    weights = kwargs.get('weights', [1])
    
    # Prepare data arrays
    processed_arrays = []
    
    for grid_data in grid_points:
        grid_data = np.asarray(grid_data)
        # Reshape to (time, features)
        grid_data = grid_data.reshape((365, -1), order='F')
        
        # Check for NaN values
        if np.isnan(grid_data).any():
            return (
                np.full(n_seas, np.nan),
                np.full(n_iter, np.nan),
                np.full(n_iter, np.nan)
            )
        
        processed_arrays.append(grid_data)
    
    # Create combined mask for valid data
    valid_mask = ~np.any([
        np.all(np.isnan(arr), axis=0) 
        for arr in processed_arrays
    ], axis=0)
    
    # Normalize and weight arrays
    normalized_arrays = []
    for arr, weight in zip(processed_arrays, weights):
        valid_data = arr[:, valid_mask]
        
        # Min-max normalization
        data_min = valid_data.min(axis=1, keepdims=True)
        data_max = valid_data.max(axis=1, keepdims=True)
        data_range = data_max - data_min
        
        # Avoid division by zero
        data_range = np.where(data_range == 0, 1, data_range)
        normalized = (valid_data - data_min) / data_range
        
        normalized_arrays.append(normalized * weight)
    
    # Combine all variables
    combined_data = np.concatenate(normalized_arrays, axis=1)
    
    # Initialize and fit RCC model
    try:
        model = RCC(
            data_to_cluster=combined_data,
            n_seas=n_seas,
            n_iter=n_iter,
            learning_rate=learning_rate,
            min_len=min_len,
            starting_bp=starting_bp
        )
        model.fit()
        
        breakpoints = model.breakpoints
        error_history = model.error_history
        prediction_history = model.prediction_history
        
    except Exception:
        return (
            np.full(n_seas, np.nan),
            np.full(n_iter, np.nan),
            np.full(n_iter, np.nan)
        )
    
    # Calculate silhouette scores
    silhouette_scores = []
    for prediction in prediction_history:
        try:
            if len(np.unique(prediction)) > 1:
                score = silhouette_score(combined_data.T, prediction)
            else:
                score = np.nan
        except Exception:
            score = np.nan
        silhouette_scores.append(score)
    
    # Ensure output arrays have correct length
    def _pad_to_length(arr: List, target_length: int) -> np.ndarray:
        """Pad array to target length with last value."""
        arr = list(arr)
        if len(arr) < target_length:
            arr.extend([arr[-1]] * (target_length - len(arr)))
        return np.array(arr[:target_length], dtype=float)
    
    error_history = _pad_to_length(error_history, n_iter)
    silhouette_scores = _pad_to_length(silhouette_scores, n_iter)
    
    return (
        np.array(breakpoints, dtype=float),
        error_history,
        silhouette_scores
    )


def XRCC(
    datasets: List[xr.DataArray], 
    n_seas: int = 2,
    iters: int = 20,
    learning_rate: int = 10,
    min_len: int = 30,
    starting_bp: Optional[List[int]] = None,
    weights: Optional[List[float]] = None,
    **kwargs
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Apply RCC clustering to xarray datasets.
    
    Parameters
    ----------
    datasets : List[xr.DataArray]
        List of input datasets to be clustered.
    n_seas : int, default=2
        Number of seasons to identify.
    iters : int, default=20
        Number of optimization iterations.
    learning_rate : int, default=10
        Maximum perturbation for breakpoint updates.
    min_len : int, default=30
        Minimum season length in days.
    starting_bp : Optional[List[int]], default=None
        Initial breakpoints.
    weights : Optional[List[float]], default=None
        Weights for each variable.
    **kwargs
        Additional parameters (mode is ignored for backward compatibility).
        
    Returns
    -------
    Tuple[xr.DataArray, xr.DataArray, xr.DataArray]
        Breakpoints, error history, and silhouette scores as DataArrays.
    """
    if weights is None:
        weights = [1.0] * len(datasets)
    
    # Remove unsupported parameters for backward compatibility
    kwargs.pop('mode', None)
    
    # Prepare parameters
    cluster_params = {
        'n_seas': n_seas,
        'iters': iters,
        'learning_rate': learning_rate,
        'min_len': min_len,
        'starting_bp': starting_bp,
        'weights': weights,
        **kwargs
    }
    
    # Apply clustering using xarray.apply_ufunc
    result = xr.apply_ufunc(
        _cluster_gridpoint,
        *datasets,
        kwargs=cluster_params,
        input_core_dims=[['time']] * len(datasets),
        output_core_dims=[['cluster'], ['iter'], ['iter']],
        vectorize=True,
        dask='parallelized',  # Fixed: was 'parallelize'
        output_dtypes=[float, float, float],
        output_sizes={'cluster': n_seas, 'iter': iters}
    )
    
    breakpoints, error_history, silhouette_scores = result
    
    # Create properly named DataArrays
    coords = {
        'lat': datasets[0].lat,
        'lon': datasets[0].lon
    }
    
    breakpoints_da = xr.DataArray(
        breakpoints,
        dims=['lat', 'lon', 'cluster'],
        coords={**coords, 'cluster': range(n_seas)},
        name='breakpoints',
        attrs={'description': 'Seasonal breakpoints in day of year'}
    )
    
    error_history_da = xr.DataArray(
        error_history,
        dims=['lat', 'lon', 'iter'],
        coords={**coords, 'iter': range(iters)},
        name='error_history',
        attrs={'description': 'RCC optimization error history'}
    )
    
    silhouette_scores_da = xr.DataArray(
        silhouette_scores,
        dims=['lat', 'lon', 'iter'],
        coords={**coords, 'iter': range(iters)},
        name='silhouette_scores',
        attrs={'description': 'Silhouette score optimization history'}
    )
    
    return breakpoints_da, error_history_da, silhouette_scores_da
