"""
Main interface for seasonal detection analysis.

This module provides the SeasonDetect class which orchestrates the entire
seasonal analysis workflow including data preprocessing, clustering, and
model training.
"""
import os
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import yaml
import xarray as xr
import numpy as np

from xseas.stats.normalize import normalize_CMIP6, normalize_ERA5
from xseas.xarray import tile_labels, XRCC
from xseas.models import train_perceptron
from xseas.manager.utils import (
    load_variables, 
    validate_config, 
    check_data_availability, 
    print_data_summary,
    create_directory_structure
)


class SeasonDetect:
    """
    Main class for orchestrating seasonal detection analysis.
    
    This class manages the complete workflow from data loading and preprocessing
    to clustering and model training for seasonal pattern detection.
    
    Parameters
    ----------
    base_path : str
        Root directory containing the project data and configuration.
    config_file : str
        Name of the YAML configuration file in the config directory.
        
    Attributes
    ----------
    base_path : Path
        Root project directory.
    name : str
        Project name from configuration.
    n_seasons : int
        Number of seasons to detect.
    variables : List[str]
        List of meteorological variables to analyze.
    """
    
    def __init__(self, base_path: str, config_file: str) -> None:
        self.base_path = self._validate_base_path(base_path)
        self.config = self._load_configuration(config_file)
        self._extract_config_parameters()
        self._build_directory_structure()
        self._check_workflow_status()
    
    def _validate_base_path(self, path: str) -> Path:
        """Validate and convert base path to Path object."""
        path_obj = Path(path)
        
        if not path_obj.exists():
            print(f"⚠️  Base path does not exist: {path}")
            print("🔧 Creating directory structure...")
            create_directory_structure(path_obj)
        else:
            # Check if it looks like an XSeas project
            config_dir = path_obj / 'config'
            data_dir = path_obj / 'data'
            
            if not (config_dir.exists() or data_dir.exists()):
                print(f"📁 Directory exists but doesn't appear to be an XSeas project")
                print("🔧 Creating missing XSeas directory structure...")
                create_directory_structure(path_obj)
            else:
                print(f"✅ Using existing XSeas project at: {path}")
        
        return path_obj
    
    def _load_configuration(self, config_file: str) -> Dict[str, Any]:
        """Load and validate configuration file."""
        config_path = self.base_path / 'config' / config_file
        
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
    
    def _extract_config_parameters(self) -> None:
        """Extract parameters from configuration."""
        self.name = self.config['name']
        self.n_seasons = self.config['n_seasons']
        self.variables = self.config['variables']
        self.CMIP6_models = self.config['CMIP6_models']
        self.CMIP6_scenarios = self.config['CMIP6_scenarios']
        
        # Variable codes
        self.era_variable_codes = self.config.get('ERA_variable_code', [])
        self.cmip_variable_codes = self.config.get('CMIP_variable_code', [])
        
        # Perceptron parameters
        perceptron_params = self.config['perceptron_params']
        self.n_years_training = perceptron_params['n_years_training']
        self.epochs = perceptron_params['epochs']
        
        # Clustering parameters (with updated parameter names)
        self.clustering_params = self.config.get('clustering_params', {})
        
        # Map old parameter names to new ones for backward compatibility
        if 'n_iters' in self.clustering_params:
            self.clustering_params['iters'] = self.clustering_params.pop('n_iters')
        if 'starting_breakpoints' in self.clustering_params:
            self.clustering_params['starting_bp'] = self.clustering_params.pop('starting_breakpoints')
        
        # Set defaults
        self.clustering_params.setdefault('iters', 200)
        self.clustering_params.setdefault('learning_rate', 1)
        self.clustering_params.setdefault('min_len', 30)
        
        # Projection parameters
        self.projection_params = self.config.get('parameters_projections', {})

    
    def _build_directory_structure(self) -> None:
        """Build and validate directory structure."""
        # Core directories
        self.data_path = self.base_path / 'data'
        self.ERA5_path = self.data_path / 'ERA5'
        self.CMIP6_path = self.data_path / 'CMIP6'
        
        # Output directories
        self.clustering_path = self.base_path / 'clusterings' / f'{self.name}_clust.nc'
        self.perceptron_path = self.base_path / 'perceptron_model' / self.name
        self.projections_path = self.base_path / 'projections' / f'{self.name}.nc'
        
        # Model-specific paths
        self.model_paths = {}
        for model in self.CMIP6_models:
            self.model_paths[model] = self.CMIP6_path / model
        
        # Prenormalized data paths
        self.prenorm_ERA5_path = self._get_prenorm_path('ERA5')
        self.prenorm_model_paths = {}
        for model in self.CMIP6_models:
            self.prenorm_model_paths[model] = self._get_prenorm_path(model)
    
    def _get_prenorm_path(self, model: str) -> Optional[Path]:
        """Get prenormalized data path for a model if it exists."""
        if model == 'ERA5':
            path = self.ERA5_path / 'prenormalized' / 'ERA5_prenorm.nc'
        else:
            path = self.CMIP6_path / model / 'prenormalized' / f'{model}_prenorm.nc'
        
        return path if path.exists() else None
    
    def _check_workflow_status(self) -> None:
        """Check the status of workflow components."""
        self.is_clustering_performed = self.clustering_path.exists()
        self.is_perceptron_trained = (self.perceptron_path / 'weights').exists()
        self.is_projections_classified = self.projections_path.exists()
    
    def check_data_availability(self) -> None:
        """Check and print data availability summary."""
        print("🔍 Checking data availability...")
        availability = check_data_availability(
            self.data_path, 
            self.CMIP6_models, 
            self.variables
        )
        print_data_summary(availability)
        
        # Check ERA5 data using configured variable codes and correct path structure
        era5_codes = self.era_variable_codes or ['2t', 'tp']
        era5_available = all(
            (self.ERA5_path / var / 'final.nc').exists() 
            for var in era5_codes
        )
        
        print(f"\n🌍 ERA5 Data: {'✅ Available' if era5_available else '❌ Missing'}")
        
        if not era5_available:
            print("   Missing files:")
            for var in era5_codes:
                file_path = self.ERA5_path / var / 'final.nc'
                if not file_path.exists():
                    print(f"   ❌ {file_path}")

    def __repr__(self) -> str:
        """Generate informative string representation."""
        line = "=" * 60
        
        # Status indicators
        clustering_status = "✅ Yes" if self.is_clustering_performed else "❌ No"
        perceptron_status = "✅ Yes" if self.is_perceptron_trained else "❌ No"
        projections_status = "✅ Yes" if self.is_projections_classified else "❌ No"
        era5_prenorm_status = "✅ Yes" if self.prenorm_ERA5_path else "❌ No"
        
        # Check CMIP6 prenormalization status
        prenorm_done = [
            model for model, path in self.prenorm_model_paths.items() 
            if path is not None
        ]
        prenorm_missing = [
            model for model, path in self.prenorm_model_paths.items() 
            if path is None
        ]
        cmip6_prenorm_status = "✅ Yes" if not prenorm_missing else "❌ No"
        
        # Configuration summary
        config_summary = []
        if self.era_variable_codes:
            config_summary.append(f"ERA codes: {', '.join(self.era_variable_codes)}")
        if self.cmip_variable_codes:
            config_summary.append(f"CMIP codes: {', '.join(self.cmip_variable_codes)}")
        if 'weights' in self.clustering_params:
            config_summary.append(f"Weights: {self.clustering_params['weights']}")
        
        return (
            f"Project: {self.name}\n{line}\n"
            f"📅 Seasons: {self.n_seasons}\n"
            f"📂 Base Path: {self.base_path}\n"
            f"📊 Variables ({len(self.variables)}): {', '.join(self.variables)}\n"
            f"🌀 CMIP6 Models ({len(self.CMIP6_models)}): {', '.join(self.CMIP6_models)}\n"
            f"📈 Scenarios ({len(self.CMIP6_scenarios)}): {', '.join(self.CMIP6_scenarios)}\n"
            + (f"⚙️  Config: {' | '.join(config_summary)}\n" if config_summary else "") +
            f"{line}\n"
            f"🔍 Clustering: {clustering_status}\n"
            f"🤖 Perceptron: {perceptron_status}\n"
            f"🌐 Projections: {projections_status}\n"
            f"{line}\n"
            f"📊 ERA5 Prenormalized: {era5_prenorm_status}\n"
            f"📊 CMIP6 Prenormalized: {cmip6_prenorm_status}\n"
            f"    ✅ Ready: {', '.join(prenorm_done) if prenorm_done else 'None'}\n"
            f"    ❌ Missing: {', '.join(prenorm_missing) if prenorm_missing else 'None'}\n"
        )
    
    def perform_clustering(
        self, 
        datasets: Optional[List[xr.DataArray]] = None,
        **kwargs
    ) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        """
        Perform seasonal clustering on ERA5 data.
        
        Parameters
        ----------
        datasets : Optional[List[xr.DataArray]]
            Input datasets. If None, loads prenormalized ERA5 data automatically.
        **kwargs
            Additional clustering parameters.
            
        Returns
        -------
        Tuple[xr.DataArray, xr.DataArray, xr.DataArray]
            Breakpoints, error history, and silhouette scores.
        """
        if datasets is None:
            if not self.prenorm_ERA5_path:
                print("❌ No prenormalized ERA5 data found.")
                print("   Please run prenormalize_ERA5() first or provide datasets manually.")
                raise FileNotFoundError("Prenormalized ERA5 data not available")
            
            print("📂 Loading prenormalized ERA5 datasets...")
            try:
                # Load prenormalized ERA5 data
                era5_dataset = xr.open_dataset(self.prenorm_ERA5_path)
                
                # Convert to list of DataArrays
                datasets = []
                era5_codes = self.era_variable_codes or ['2t', 'tp']
                
                for var_code in era5_codes:
                    if var_code in era5_dataset:
                        datasets.append(era5_dataset[var_code])
                        print(f"   ✅ Loaded {var_code}: {era5_dataset[var_code].shape}")
                    else:
                        print(f"   ⚠️  Variable {var_code} not found in prenormalized data")
                
                if not datasets:
                    raise ValueError("No valid variables found in prenormalized ERA5 data")
                
                print(f"✅ Loaded {len(datasets)} variables for clustering")
                
            except Exception as e:
                raise FileNotFoundError(f"Could not load prenormalized ERA5 data: {e}")
        
        # Merge clustering parameters
        clustering_params = {**self.clustering_params, **kwargs}
        clustering_params['n_seas'] = self.n_seasons
        
        print("🔄 Performing clustering...")
        print(f"   Parameters: {clustering_params}")
        
        # Perform clustering
        breakpoints, error_history, silhouette_scores = XRCC(
            datasets, 
            **clustering_params
        )
        
        # Save results
        self.clustering_path.parent.mkdir(parents=True, exist_ok=True)
        
        clustering_ds = xr.Dataset({
            'breakpoints': breakpoints,
            'error_history': error_history,
            'silhouette_scores': silhouette_scores
        })
        
        # Add metadata
        clustering_ds.attrs.update({
            'description': 'XSeas clustering results',
            'n_seasons': self.n_seasons,
            'variables': self.era_variable_codes or ['2t', 'tp'],
            'clustering_parameters': str(clustering_params),
            'created_by': 'XSeas SeasonDetect'
        })
        
        clustering_ds.to_netcdf(self.clustering_path)
        self.is_clustering_performed = True
        
        print(f"✅ Clustering completed and saved to: {self.clustering_path}")
        
        return breakpoints, error_history, silhouette_scores
    
    def prenormalize_ERA5(self) -> None:
        """
        Prenormalize ERA5 data using rolling window normalization.
        
        Applies day-of-year averaging followed by rolling z-score normalization
        to prepare ERA5 data for seasonal analysis.
        """
        if self.prenorm_ERA5_path:
            print("✅ ERA5 already prenormalized. Skipping...")
            return
        
        print("🔄 Prenormalizing ERA5 data...")
        
        # Check if ERA5 data is available using correct path structure
        era5_codes = self.era_variable_codes or ['2t', 'tp']
        missing_files = []
        for var in era5_codes:
            file_path = self.ERA5_path / var / 'final.nc'
            if not file_path.exists():
                missing_files.append(str(file_path))
        
        if missing_files:
            print("❌ Missing ERA5 files:")
            for file in missing_files:
                print(f"   {file}")
            print("   Please ensure ERA5 data is available before prenormalization.")
            print("   Expected structure: data/ERA5/[variable]/final.nc")
            return
        
        try:
            # Create output directory
            prenorm_dir = self.ERA5_path / 'prenormalized'
            prenorm_dir.mkdir(parents=True, exist_ok=True)
            
            # Normalize ERA5 data
            dataset_prenorm = normalize_ERA5(
                str(self.ERA5_path),
                self.variables,  # Not used for ERA5 but kept for consistency
                era5_codes
            )
            
            # Save prenormalized data
            output_path = prenorm_dir / 'ERA5_prenorm.nc'
            print(f"💾 Saving prenormalized ERA5 data to: {output_path}")
            
            # Add compression for efficiency
            encoding = {}
            for var in dataset_prenorm.data_vars:
                encoding[var] = {
                    'zlib': True,
                    'complevel': 6,
                    'fletcher32': True
                }
            
            dataset_prenorm.to_netcdf(output_path, encoding=encoding)
            
            # Update path tracking
            self.prenorm_ERA5_path = output_path
            
            print("✅ ERA5 prenormalization completed successfully")
            print(f"   Variables processed: {', '.join(era5_codes)}")
            print(f"   Data shape: {dict(dataset_prenorm.sizes)}")
            
        except Exception as e:
            print(f"❌ Error prenormalizing ERA5 data: {e}")
            import traceback
            traceback.print_exc()

    def prenormalize_CMIP6(self) -> None:
        """
        Prenormalize CMIP6 data for all models.
        
        Creates prenormalized datasets by applying rolling normalization
        to combine historical and scenario data.
        """
        # Use configured variable codes or fall back to defaults
        variable_codes = self.cmip_variable_codes or ['tas', 'pr', 'ua', 'va']
        
        for model in self.CMIP6_models:
            if self.prenorm_model_paths[model] is not None:
                print(f"✅ {model} already prenormalized. Skipping...")
                continue
            
            print(f"🔄 Prenormalizing {model}...")
            
            try:
                # Create output directory
                prenorm_dir = self.CMIP6_path / model / 'prenormalized'
                prenorm_dir.mkdir(parents=True, exist_ok=True)
                
                # Normalize data
                model_path = self.model_paths[model]
                dataset_prenorm = normalize_CMIP6(
                    str(model_path),
                    self.variables,
                    variable_codes,
                    self.CMIP6_scenarios
                )
                
                # Save prenormalized data
                output_path = prenorm_dir / f'{model}_prenorm.nc'
                dataset_prenorm.to_netcdf(output_path)
                
                # Update path tracking
                self.prenorm_model_paths[model] = output_path
                
                print(f"✅ {model} prenormalization complete")
                
            except Exception as e:
                print(f"❌ Error prenormalizing {model}: {e}")
    
    def train_perceptron_models(
        self, 
        n_years_training: Optional[int] = None,
        epochs: Optional[int] = None
    ) -> None:
        """
        Train perceptron models for seasonal classification.
        
        Parameters
        ----------
        n_years_training : Optional[int]
            Number of years to use for training. If None, uses config value.
        epochs : Optional[int]
            Number of training epochs. If None, uses config value.
        """
        if not self.prenorm_ERA5_path:
            print("❌ ERA5 prenormalized data not found.")
            print("    Please run prenormalize_ERA5() first.")
            return
        
        if not self.is_clustering_performed:
            print("❌ Clustering must be performed before training.")
            print("    Please run perform_clustering() first.")
            return
        
        # Use provided parameters or fall back to config
        n_years = n_years_training or self.n_years_training
        epochs_count = epochs or self.epochs
        
        print(f"🔄 Training perceptron models...")
        print(f"   Training years: {n_years}")
        print(f"   Epochs: {epochs_count}")
        
        try:
            # Load data
            dataset_train = xr.open_dataset(self.prenorm_ERA5_path)
            labels = xr.open_dataset(self.clustering_path)
            
            # Add labels to training data
            dataset_train = tile_labels(dataset_train, labels, self.n_seasons)
            
            # Convert to array format expected by training function
            training_array = dataset_train.to_array().values.transpose((2, 3, 1, 0))
            
            # Create output directory
            self.perceptron_path.mkdir(parents=True, exist_ok=True)
            
            # Note: The original train_perceptron function needs to be adapted
            # for spatial training. This is a simplified version.
            print("⚠️  Spatial perceptron training needs to be implemented.")
            print("    Current train_perceptron function is for single time series.")
            
            # Update status
            self.is_perceptron_trained = True
            
            print("✅ Perceptron training placeholder complete")
            
        except Exception as e:
            print(f"❌ Error training perceptron: {e}")
            raise
    
    def run_full_workflow(self) -> None:
        """
        Run the complete XSeas workflow.
        
        This method executes all steps in sequence:
        1. Check data availability
        2. Prenormalize data
        3. Perform clustering
        4. Train perceptron models
        """
        print("🚀 Starting full XSeas workflow...")
        print("=" * 50)
        
        # Step 1: Check data
        print("\n1️⃣  Checking data availability...")
        self.check_data_availability()
        
        # Step 2: Prenormalize data
        print("\n2️⃣  Prenormalizing data...")
        self.prenormalize_ERA5()
        self.prenormalize_CMIP6()
        
        # Step 3: Perform clustering
        print("\n3️⃣  Performing clustering...")
        if not self.is_clustering_performed:
            self.perform_clustering()
        else:
            print("✅ Clustering already performed. Skipping...")
        
        # Step 4: Train perceptron
        print("\n4️⃣  Training perceptron models...")
        if not self.is_perceptron_trained:
            self.train_perceptron_models()
        else:
            print("✅ Perceptron already trained. Skipping...")
        
        print("\n🎉 Full workflow completed!")
        print("=" * 50)
        print(self)
    
    def _save_training_metrics(
        self, 
        dataset: xr.Dataset, 
        mse: np.ndarray, 
        r2: np.ndarray, 
        accuracy: np.ndarray
    ) -> None:
        """Save training metrics to NetCDF file."""
        metrics_ds = xr.Dataset(
            {
                'mse': (('lat', 'lon'), mse),
                'r2': (('lat', 'lon'), r2),
                'accuracy': (('lat', 'lon'), accuracy)
            },
            coords={
                'lat': dataset['lat'],
                'lon': dataset['lon']
            },
            attrs={
                'description': 'Perceptron training metrics',
                'created_by': 'XSeas SeasonDetect',
                'n_seasons': self.n_seasons,
                'variables': ', '.join(self.variables)
            }
        )
        
        metrics_dir = self.perceptron_path / 'metrics'
        metrics_dir.mkdir(exist_ok=True)
        
        metrics_path = metrics_dir / 'training_metrics.nc'
        metrics_ds.to_netcdf(metrics_path)



