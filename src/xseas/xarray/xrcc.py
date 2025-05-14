import numpy as np
import xarray as xr
from xseas.models import RCC 
from sklearn.metrics import silhouette_score


def X_cluster(*grid_points, **kwargs):

    iters = kwargs.get('iters', 20)  
    n_seas = kwargs.get('n_seas', 2)
    learning_rate = kwargs.get('learning_rate', 10)
    min_len = kwargs.get('min_len', 30)
    starting_bp = kwargs.get('starting_bp', None)
    weights = kwargs.get('weights', [1])

    arrays = []
    
    for grid_points_var in grid_points:
        grid_points_var = np.asarray(grid_points_var)
        grid_points_var = np.reshape(grid_points_var, (365, int(grid_points_var.size/365)), order='F')
        
        if np.isnan(grid_points_var).any():
 
            return (np.full(n_seas, np.nan), 
                    np.full(iters, np.nan), 
                    np.full(iters, np.nan)) 
        
        arrays.append(grid_points_var)
    
    combined_mask = ~np.any([np.all(np.isnan(arr), axis=0) for arr in arrays], axis=0)
    
    normalized_arrays = []
    for arr, weight in zip(arrays, weights):
        array_tot = arr[:, combined_mask]
        array_tot = (array_tot - array_tot.min(axis=1).reshape(-1, 1)) / (array_tot.max(axis=1) - array_tot.min(axis=1)).reshape(-1, 1)
        normalized_arrays.append(array_tot * weight)
    
    array_tot = np.concatenate(normalized_arrays, axis=1)

    # Initialize and fit the Radially_Constrained_Cluster model
    model = RCC(    data_to_cluster=array_tot,
                    n_seas=n_seas,
                    n_iter=iters,
                    learning_rate=learning_rate,
                    min_len=min_len,
                    starting_bp=starting_bp
                    )
    model.fit()

    breakpoints = model.breakpoints
    breakpoint_history = model.breakpoint_history
    error_history = model.error_history
    prediction_history = model.prediction_history

    silhouette_scores = []
    for pred in prediction_history:
        try:
            score = silhouette_score(array_tot, pred)
            silhouette_scores.append(score)
        except:
            silhouette_scores.append(np.nan)

    #print(len(prediction_history), len(silhouette_scores), len(error_history))

    prediction_history = prediction_history.tolist()
    error_history = error_history.tolist()

    if len(prediction_history) != iters:
        prediction_history.extend([prediction_history[-1]] * (iters - len(prediction_history)))
        silhouette_scores.extend([silhouette_scores[-1]] * (iters - len(silhouette_scores)))
        error_history.extend([error_history[-1]] * (iters - len(error_history)))

    return (
        np.array(breakpoints), 
        np.array(error_history, dtype=float), 
        np.array(silhouette_scores, dtype=float)
    )



def XRCC(datasets, **kwargs):
    """
    Apply clustering function to a list of xarray DataArrays.

    Parameters:
    -----------
    datasets : list of xarray DataArrays
        Input datasets to be clustered.
    **kwargs : keyword arguments
        Additional parameters for clustering.

    Returns:
    --------
    tuple of xarray.DataArrays
        Result of clustering operation including breakpoints, 
        error history, and silhouette scores.
    """

    n_seas = kwargs.get('n_seas', 2)
    iters = kwargs.get('iters', 20)

    result = xr.apply_ufunc(
        X_cluster,
        *datasets,
        kwargs=kwargs,
        input_core_dims=[['time']] * len(datasets),
        output_core_dims=[['cluster'] ,['iter'], ['iter']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float, float, float],
        output_sizes={'cluster': n_seas, 'iter': iters}
    )

    breakpoints, error_history, silhouette_scores = result

    breakpoints = xr.DataArray(breakpoints, dims=['lat', 'lon', 'cluster'])
    error_history_da = xr.DataArray(error_history, dims=['lat', 'lon','iter'])
    silhouette_scores_da = xr.DataArray(silhouette_scores, dims=['lat', 'lon', 'iter'])


    return breakpoints, error_history_da, silhouette_scores_da
