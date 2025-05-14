import numpy as np
import xarray as xr
import warnings
from tqdm import tqdm
import argparse
import os
import yaml

from xseas.utils import load_variables

# Suppress warnings
warnings.filterwarnings("ignore")



def rolling_doy_complete(da, window_size=30):

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




def normalize_CMIP6(base_path, variables, variables_codes, scenarios):

    normalized_dataset = []

    for scenario in scenarios:

        dataset = load_variables(os.path.join(base_path, scenario), variables, variables_codes)

        for dat in dataset:
            normalized_dataset.append(dat)

    dataset_proj = xr.merge(normalized_dataset).load()
    dataset_proj = rolling_doy_complete(dataset_proj, window_size=15)
    dataset_proj = rolling_zscore_complete(dataset_proj, window_size=365*10)
    
    return dataset_proj