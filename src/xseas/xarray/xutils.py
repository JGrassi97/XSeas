import numpy as np
import xarray as xr


def get_prediction(b, **kwargs):
    n_seas = kwargs['n_seas']
    prediction = np.zeros(365)

    try:
        idx = generate_season_idx(b, n_seas)
        for i in range(n_seas):
            prediction[idx[i].astype(int)] = i

        return prediction.astype(int)

    except:
        return np.nan * np.ones(365)



def generate_season_idx(b, n_seas):
    idx = []

    if n_seas == 1:
        idx.append(np.arange(0, 365, 1))

    else:
        for i in np.arange(-1, n_seas-1,1):
            if b[i]>b[i+1]:
                idx_0 = np.arange(b[i], 365, 1)
                idx_1 = np.arange(0, b[i+1], 1)
                idx.append(np.concatenate((idx_0, idx_1), axis=None))

            else:
                idx.append(np.arange(b[i], b[i+1],1))

    return idx



def X_labels(breakpoints: xr.DataArray, **kwargs):
    dates_clust = xr.apply_ufunc(
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
    return dates_clust



def tile_labels(dataset, labels, n_seasons):
    
    lables_param = {'n_seas':n_seasons }
    dates_clust = X_labels(labels['breakpoints'], **lables_param)

    # Compute the number of years in the dataset
    n_years = dataset.resample(time='1Y')._len

    index_values = dates_clust.values
    index_values = np.tile(index_values, n_years).transpose((2, 0, 1))
    dataset['labels'] = (('time', 'lat', 'lon'), index_values)

    return dataset