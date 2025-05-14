import os
import xarray as xr

# -- UTILITY FUNCTIONS FOR LOADING DATA -- #
# This function loads the variables from the specified base path and returns a list of datasets.

def load_variables(base_path : str[str], 
                   variables : list[str],
                   variables_codes : list) -> list[xr.DataArray]:
    """ Load variables from the specified base path and return a list of xr.DataArray.
        This utility is mainly intended for optimizing the loading of CMIP6 data.
        TODO: remove this utility and improve the data ingestion process """
    
    dataset = []
    for variable, code in zip(variables, variables_codes):
            path = os.path.join(base_path, variable, 'final.nc')

            try:
                dat = xr.open_dataset(path)[code].mean('plev')
            except:
                dat =xr.open_dataset(path)[code]

            dataset.append(dat)

    return dataset