"""
Utility functions for xarray operations in seasonal analysis.

This module provides functions for generating seasonal labels and predictions
from breakpoint data using xarray's apply_ufunc capabilities.
"""
from typing import Dict, Any
import numpy as np
import xarray as xr


def get_prediction(b, **kwargs):
    n_seas = kwargs['n_seas']
    prediction = np.zeros(365)

    try:
        idx = _generate_season_indices(b, n_seas)
        for i in range(n_seas):
            prediction[idx[i].astype(int)] = i

        return prediction.astype(int)

    except:
        return np.nan * np.ones(365)



def _generate_season_indices(b, n_seas):
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



def generate_labels(breakpoints: xr.DataArray, **kwargs):
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
    """Aggiunge le etichette stagionali all'intero asse temporale del dataset.

    Genera le etichette (365 valori per anno) a partire dai breakpoint e le
    ripete lungo la dimensione *time* del dataset, gestendo:
      - Dataset con numero di giorni multiplo/non multiplo di 365
      - Presenza di anni bisestili (elimina il giorno 366 se esiste)
      - Breakpoint mancanti (inserisce NaN)
    """
    # Parametri per generare le etichette giornaliere (0..n_seasons-1)
    label_params = {'n_seas': n_seasons}
    bp = labels['breakpoints']

    # Genera labels shape: (lat, lon, dayofyear=365)
    dates_clust = generate_labels(bp, **label_params)

    # Base year pattern (365, lat, lon)
    base_year = dates_clust.transpose('dayofyear', 'lat', 'lon').values  # (365, lat, lon)

    time = dataset['time']
    time_len = time.size

    # Gestione anni bisestili nel dataset: se esistono DOY=366 lo rimuoviamo
    # Calcoliamo dayofyear (xarray gestisce correttamente calendari standard)
    if 'dayofyear' in time.dt.__dir__():
        dayofyear = time.dt.dayofyear.values
    else:
        # fallback: costruiamo sequenza semplice
        dayofyear = (np.arange(time_len) % 365) + 1

    # Conteggio anni stimato come floor
    n_full_years = time_len // 365
    remainder = time_len % 365

    # Ripetiamo il pattern per il numero totale necessario (incluso remainder)
    repeat_needed = n_full_years + (1 if remainder else 0)
    tiled = np.tile(base_year, (repeat_needed, 1, 1))  # (repeat_needed*365, lat, lon)

    # Se abbiamo leap days nel dataset (DOY=366) li mappiamo al giorno 365
    # Creiamo array finale rispettando l'ordine temporale originale
    labels_time = np.empty((time_len, base_year.shape[1], base_year.shape[2]), dtype=base_year.dtype)

    # Indici globali per il pattern ripetuto
    for i in range(time_len):
        doy = dayofyear[i]
        if doy == 366:
            doy = 365  # Collassa giorno 366 su 365
        # Posizione nel pattern ripetuto
        year_index = (i // 365)
        day_index = doy - 1  # 0-based
        labels_time[i] = tiled[year_index * 365 + day_index]

    # Assegna (time, lat, lon)
    dataset = dataset.copy()
    dataset['labels'] = (('time', 'lat', 'lon'), labels_time)

    return dataset




# Backward compatibility
X_labels = generate_labels  # Alias for backward compatibility