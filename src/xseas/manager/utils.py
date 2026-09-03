"""
Utility functions for the manager module.

This module provides utility functions for data loading, path management,
and configuration handling in the XSeas workflow.
"""
from typing import List, Optional, Union, Dict, Any
import os
import yaml
from pathlib import Path
import xarray as xr
import numpy as np


def load_variables(
    base_path: str, 
    variables: List[str],
    variables_codes: List[str]
) -> List[xr.DataArray]:
    """
    Load variables from the specified base path and return a list of xr.DataArray.
    
    This utility is mainly intended for optimizing the loading of CMIP6 data.
    
    Parameters
    ----------
    base_path : str
        Base path containing variable directories.
    variables : List[str]
        List of variable names (directory names).
    variables_codes : List[str]
        List of variable codes in NetCDF files.
        
    Returns
    -------
    List[xr.DataArray]
        List of loaded DataArrays.
    """
    datasets = []
    
    for variable, code in zip(variables, variables_codes):
        path = os.path.join(base_path, variable, 'final.nc')
        
        if not os.path.exists(path):
            print(f"⚠️  Warning: File not found: {path}")
            continue
            
        try:
            # Try to load with pressure level averaging
            dat = xr.open_dataset(path)[code].mean('plev')
        except KeyError:
            try:
                # Load without pressure level
                dat = xr.open_dataset(path)[code]
            except KeyError:
                print(f"⚠️  Warning: Variable code '{code}' not found in {path}")
                continue
        except Exception as e:
            print(f"⚠️  Warning: Error loading {path}: {e}")
            continue
            
        datasets.append(dat)
    
    return datasets


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary to validate.
        
    Returns
    -------
    bool
        True if valid, False otherwise.
    """
    required_keys = [
        'name', 'n_seasons', 'variables', 'CMIP6_models', 
        'CMIP6_scenarios', 'perceptron_params'
    ]
    
    # Check required top-level keys
    for key in required_keys:
        if key not in config:
            print(f"❌ Missing required key: {key}")
            return False
    
    # Check optional but recommended keys
    recommended_keys = [
        'ERA_variable_code', 'CMIP_variable_code', 
        'clustering_params', 'parameters_projections'
    ]
    
    for key in recommended_keys:
        if key not in config:
            print(f"⚠️  Recommended key missing: {key}")
    
    # Validate perceptron_params
    perceptron_required = ['n_years_training', 'epochs']
    for key in perceptron_required:
        if key not in config['perceptron_params']:
            print(f"❌ Missing required perceptron parameter: {key}")
            return False
    
    # Validate clustering_params if present
    if 'clustering_params' in config:
        clustering_recommended = ['n_iters', 'learning_rate', 'min_len']
        for key in clustering_recommended:
            if key not in config['clustering_params']:
                print(f"⚠️  Recommended clustering parameter missing: {key}")
    
    # Validate data types
    if not isinstance(config['n_seasons'], int) or config['n_seasons'] < 1:
        print("❌ n_seasons must be a positive integer")
        return False
        
    if not isinstance(config['variables'], list) or len(config['variables']) == 0:
        print("❌ variables must be a non-empty list")
        return False
    
    # Validate variable codes alignment
    if 'ERA_variable_code' in config and 'CMIP_variable_code' in config:
        if len(config['variables']) != len(config['ERA_variable_code']):
            print("⚠️  Number of variables doesn't match ERA variable codes")
        if len(config['variables']) != len(config['CMIP_variable_code']):
            print("⚠️  Number of variables doesn't match CMIP variable codes")
    
    return True


def create_directory_structure(base_path: Path) -> None:
    """
    Create the standard XSeas directory structure.
    
    Parameters
    ----------
    base_path : Path
        Base path where to create the structure.
    """
    directories = [
        'data/ERA5/prenormalized',
        'data/CMIP6',
        'config',
        'clusterings',
        'perceptron_model',
        'projections',
        'results',
        'notebooks'
    ]
    
    created_count = 0
    for directory in directories:
        dir_path = base_path / directory
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created directory: {dir_path}")
            created_count += 1
        else:
            print(f"✅ Directory already exists: {dir_path}")
    
    if created_count == 0:
        print("✅ All required directories already exist")
    else:
        print(f"📁 Created {created_count} new directories")


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load and validate YAML configuration file.
    
    Parameters
    ----------
    config_path : Union[str, Path]
        Path to configuration file.
        
    Returns
    -------
    Dict[str, Any]
        Configuration dictionary.
        
    Raises
    ------
    FileNotFoundError
        If configuration file doesn't exist.
    ValueError
        If configuration is invalid.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML configuration: {e}")
    
    if not validate_config(config):
        raise ValueError("Configuration validation failed")
    
    return config


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary to save.
    config_path : Union[str, Path]
        Path where to save the configuration.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False, indent=2)
    
    print(f"💾 Configuration saved to: {config_path}")


def check_data_availability(
    data_path: Path, 
    models: List[str], 
    variables: List[str]
) -> Dict[str, Dict[str, bool]]:
    """
    Check availability of data files for given models and variables.
    
    Parameters
    ----------
    data_path : Path
        Base data path.
    models : List[str]
        List of model names to check.
    variables : List[str]
        List of variables to check.
        
    Returns
    -------
    Dict[str, Dict[str, bool]]
        Nested dictionary with availability status.
    """
    availability = {}
    
    for model in models:
        availability[model] = {}
        model_path = data_path / 'CMIP6' / model
        
        for variable in variables:
            var_file = model_path / variable / 'final.nc'
            availability[model][variable] = var_file.exists()
    
    return availability


def print_data_summary(availability: Dict[str, Dict[str, bool]]) -> None:
    """
    Print a summary of data availability.
    
    Parameters
    ----------
    availability : Dict[str, Dict[str, bool]]
        Data availability dictionary from check_data_availability.
    """
    print("\n📊 Data Availability Summary:")
    print("=" * 50)
    
    for model, variables in availability.items():
        print(f"\n🌀 {model}:")
        for variable, available in variables.items():
            status = "✅" if available else "❌"
            print(f"   {status} {variable}")


def check_ERA5_availability(era5_path: Path, variable_codes: List[str]) -> Dict[str, bool]:
    """
    Check availability of ERA5 data files.
    
    Parameters
    ----------
    era5_path : Path
        Path to ERA5 data directory.
    variable_codes : List[str]
        List of ERA5 variable codes to check.
        
    Returns
    -------
    Dict[str, bool]
        Dictionary with availability status for each variable.
    """
    availability = {}
    
    for var_code in variable_codes:
        file_path = era5_path / f'{var_code}.nc'
        availability[var_code] = file_path.exists()
    
    return availability


def print_ERA5_summary(availability: Dict[str, bool], era5_path: Path) -> None:
    """
    Print a summary of ERA5 data availability.
    
    Parameters
    ----------
    availability : Dict[str, bool]
        ERA5 availability dictionary.
    era5_path : Path
        Path to ERA5 data directory.
    """
    print(f"\n🌍 ERA5 Data Summary ({era5_path}):")
    print("-" * 40)
    
    for var_code, available in availability.items():
        status = "✅" if available else "❌"
        print(f"   {status} {var_code}.nc")
    
    available_count = sum(availability.values())
    total_count = len(availability)
    print(f"\n   Available: {available_count}/{total_count} files")


def get_sample_config() -> Dict[str, Any]:
    """
    Get a sample configuration dictionary.
    
    Returns
    -------
    Dict[str, Any]
        Sample configuration.
    """
    return {
        'name': 'monsoon_all_vars',
        
        'variables': [
            '2m_temperature',
            'total_precipitation',
            'u850',
            'v850'
        ],
        
        'CMIP6_models': [
            'EC-Earth3',
            'ACCESS-CM2',
            'BCC-CSM2-MR',
            'CESM2-WACCM',
            'CanESM5',
            'CNRM-CM6-1',
            'FGOALS-g3',
            'GFDL-CM4',
            'IPSL-CM6A-LR',
            'MIROC6',
            'MPI-ESM1-2-HR',
            'MRI-ESM2-0',
            'UKESM1-0-LL'
        ],
        
        'CMIP6_scenarios': [
            'historical',
            'ssp585'
        ],
        
        'ERA_variable_code': [
            '2t',
            'tp',
            'u',
            'v'
        ],
        
        'CMIP_variable_code': [
            'tas',
            'pr',
            'ua',
            'va'
        ],
        
        'n_seasons': 2,
        
        'clustering_params': {
            'n_iters': 500,
            'starting_breakpoints': [160, 280],
            'learning_rate': 1,
            'scheduler': 1,
            'min_len': 10,
            'weights': [1, 1, 0.5, 0.5]
        },
        
        'perceptron_params': {
            'n_years_training': 50,
            'epochs': 100
        },
        
        'parameters_projections': {
            'scenario_model': [
                'MRI-ESM2-0'
            ],
            
            'variables': [
                '2m_temperature',
                'total_precipitation',
                'u850',
                'v850'
            ],
            
            'model_path': [
                '/home/jgrassi/work/XSeasonsDetect/INDIA/data/preprocessed/CMIP6/MRI-ESM2-0/'
            ],
            
            'variable_code': [
                'tas',
                'pr',
                'ua',
                'va',
                'tas',
                'pr',
                'ua',
                'va'
            ],
            
            'model': {
                'name': 'Perceptron',
                'params': {
                    'hidden_layer_sizes': 256,
                    'max_iter': 1000,
                    'random_state': 42
                }
            }
        }
    }


def create_sample_config(output_path: Union[str, Path]) -> None:
    """
    Create a sample configuration file.
    
    Parameters
    ----------
    output_path : Union[str, Path]
        Path where to save the sample configuration.
    """
    config = get_sample_config()
    save_config(config, output_path)
    print(f"📝 Sample configuration created at: {output_path}")